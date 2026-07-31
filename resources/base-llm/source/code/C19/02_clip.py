import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from transformers import GPT2Model, GPT2Tokenizer
import timm

# 1. 模型定义

class ImageEncoder(nn.Module):
    """图像编码器"""
    def __init__(self, output_dim):
        super(ImageEncoder, self).__init__()
        # 使用来自timm的ViT模型
        # num_classes=0 会移除分类 head，输出 backbone 特征（维度为 vit.num_features）
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0)
        self.proj = nn.Linear(self.vit.num_features, output_dim, bias=False)

    def forward(self, x):
        feat = self.vit(x)
        return self.proj(feat)


class TextEncoder(nn.Module):
    """文本编码器"""
    def __init__(self, output_dim):
        super(TextEncoder, self).__init__()
        model_name = 'gpt2'
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # GPT-2默认没有pad_token，将其设为eos_token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2Model.from_pretrained(model_name)
        self.proj = nn.Linear(self.model.config.hidden_size, output_dim, bias=False)

    def forward(self, texts):
        # 文本通过GPT-2
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(self.model.device)
        output = self.model(**inputs)
        
        # 获取最后一个非 padding token 的输出
        # 根据 attention_mask 计算每个样本的最后一个有效 token 索引
        last_hidden_state = output.last_hidden_state
        attention_mask = inputs.attention_mask
        last_token_idx = attention_mask.sum(dim=1) - 1 # (B)
        
        # 从batch中取出对应索引的向量
        batch_idx = torch.arange(last_hidden_state.size(0)).to(last_token_idx.device)
        sent = last_hidden_state[batch_idx, last_token_idx] # (B, 768)
        return self.proj(sent) # (B, output_dim)


class CLIP(nn.Module):
    """CLIP模型：结合图像和文本编码器"""
    def __init__(self, embed_dim):
        super(CLIP, self).__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(embed_dim)

        # 可学习温度系数（论文中用 log 参数化的 logit_scale）
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))

    def forward(self, images, texts):
        img = self.image_encoder(images)  # (B, embed_dim)
        txt = self.text_encoder(texts)    # (B, embed_dim)

        # 归一化后点积即余弦相似度
        img = torch.nn.functional.normalize(img, dim=-1)
        txt = torch.nn.functional.normalize(txt, dim=-1)

        # 温度缩放
        scale = self.logit_scale.exp()
        logits = scale * (img @ txt.T) # (B, B)
        return logits


# 2. 数据处理

def load_cifar10_dataset(batch_size, image_size=224, root='./cifar10', mean=None, std=None):
    """加载CIFAR10数据集"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    # download=True 会自动下载
    train_dataset = CIFAR10(root=root, train=True, download=True, transform=transform)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    classes = train_dataset.classes
    return loader, classes


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化模型
    clip_model = CLIP(embed_dim=512).to(device)
    clip_model.text_encoder.model.to(device)

    # 加载数据
    cfg = clip_model.image_encoder.vit.default_cfg
    mean = cfg['mean']
    std = cfg['std']
    data_root = os.path.join(os.path.dirname(__file__), "cifar10")
    dataset, classes = load_cifar10_dataset(batch_size=4, root=data_root, mean=mean, std=std)

    # 3. 训练循环
    for i, (images, labels) in enumerate(dataset):
        images = images.to(device)
        texts = [classes[label.item()] for label in labels]

        # 前向计算
        logits = clip_model(images, texts) # (B, B)
        
        # 计算损失
        targets = torch.arange(logits.shape[0]).to(device)
        loss_i = nn.CrossEntropyLoss()(logits, targets)
        loss_t = nn.CrossEntropyLoss()(logits.T, targets)
        loss = (loss_i + loss_t) / 2

        print(f"Batch {i}: Loss = {loss.item():.4f}")

        # 仅演示前几个batch
        if i >= 2: break
