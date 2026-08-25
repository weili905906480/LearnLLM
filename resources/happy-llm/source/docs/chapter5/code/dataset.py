"""
第五章：大语言模型训练 - 数据集处理模块

本模块定义了两个 PyTorch Dataset 类，用于大语言模型训练的不同阶段：
1. PretrainDataset：预训练阶段，对全部文本内容计算损失
2. SFTDataset：  监督微调阶段，仅对 assistant 回复部分计算损失

核心技术点：
- 惰性加载：通过预计算文件字节偏移量（offset），按需读取单行数据，避免一次性加载整个数据集到内存
- Loss Mask：控制损失计算范围，0 表示该位置不参与损失计算，1 表示参与
- 因果语言模型 (Causal LM)：使用 teacher forcing，输入 X 为 tokens[0:n-1]，标签 Y 为 tokens[1:n]
"""

import json
import random
import re
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import os


class PretrainDataset(Dataset):
    """
    预训练数据集类

    用于大语言模型的预训练阶段。预训练阶段的目标是让模型学习语言的基本规律
    （语法、语义、常识知识等），因此对输入文本的每一个有效 token 都计算损失。

    数据格式：
    - 输入文件每行为一个 JSON 对象，包含 "text" 字段
    - 示例：{"text": "这是一段用于预训练的文本内容"}

    处理流程：
    1. 从 JSON 中提取文本，添加 BOS token
    2. tokenize 后截断到 max_length
    3. 构建因果语言模型的输入输出对
    4. 生成 loss_mask（所有有效 token 位置 = 1，padding 位置 = 0）
    """

    def __init__(self, data_path, tokenizer, max_length=512):
        """
        初始化预训练数据集

        Args:
            data_path: 数据文件路径，每行一个 JSON 对象
            tokenizer: 分词器实例，需要支持 __call__ 方法返回包含 'input_ids' 的字典
            max_length: 序列最大长度，超过的部分将被截断
        """
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 获取 padding token 的 id，若未设置则默认使用 0
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        # ====== 预计算每行的起始字节偏移量（惰性加载的关键）=======
        # 原理：用二进制模式打开文件，逐行读取并记录每行开头在文件中的字节位置。
        # 后续 __getitem__ 时可以直接用 f.seek() 跳到指定位置，只读取一行，
        # 从而实现"按需加载"，避免将整个数据集读入内存。
        #
        # 文件字节位置示意：
        #   offset[0]=0    → "{"text": "..."}\n"
        #   offset[1]=100  → "{"text": "..."}\n"
        #   offset[2]=210  → [EOF]
        #
        # 注意：最后一个 offset 指向文件末尾 (EOF)，不是有效行，
        # 所以实际行数 = len(offsets) - 1
        self._offsets = []
        with open(data_path, 'rb') as f:          # 'rb' = 二进制只读模式
            self._offsets.append(0)               # 第 0 行起始于文件开头
            while f.readline():                   # 逐行读取，readline() 返回空字节表示 EOF
                self._offsets.append(f.tell())    # 记录当前文件指针位置（即下一行开头）

        # 总行数 = offset 数量 - 1（最后一个 offset 指向文件末尾）
        self._total_lines = len(self._offsets) - 1

    def __len__(self):
        """返回数据集总样本数"""
        return self._total_lines

    def __getitem__(self, index: int):
        """
        获取第 index 个样本

        返回格式：三元组 (X, Y, loss_mask)
        - X:        输入 token 序列，shape = (max_length - 1,)，取 tokens[0:n-1]
        - Y:        目标 token 序列，shape = (max_length - 1,)，取 tokens[1:n]
        - loss_mask: 损失掩码，  shape = (max_length - 1,)，1=计算损失，0=忽略

        这就是标准的"因果语言模型"训练格式：用前面的 token 预测下一个 token。

        Args:
            index: 样本索引，范围 [0, total_lines - 1]

        Returns:
            tuple: (X, Y, loss_mask)，均为 torch.Tensor (int64)
        """
        # ---- 步骤 1：高效读取单行数据 ----
        # 使用预计算的 offset 直接跳转到目标行，只读取一行而不是整个文件
        with open(self.data_path, 'rb') as f:
            f.seek(self._offsets[index])           # 跳转到第 index 行的起始位置
            line = f.readline().decode('utf-8')    # 读取一行并解码为 UTF-8 字符串

        # ---- 步骤 2：解析 JSON 并构造文本 ----
        sample = json.loads(line)                  # 解析 JSON 对象
        # 在文本开头添加 BOS (Beginning of Sentence) token
        # BOS token 标志序列开始，帮助模型识别句子边界
        text = f"{self.tokenizer.bos_token}{sample['text']}"

        # ---- 步骤 3：Tokenize 并处理长度 ----
        # 调用 tokenizer 将文本转换为数字序列，然后截断到 max_length
        input_id = self.tokenizer(text).data['input_ids'][:self.max_length]
        text_len = len(input_id)                   # 实际有效 token 数量

        # ---- 步骤 4：Padding（填充）- 确保所有样本等长 ----
        # 计算需要补充的 padding token 数量
        padding_len = self.max_length - text_len
        # 在序列尾部追加 padding token
        input_id = input_id + [self.padding] * padding_len

        # ---- 步骤 5：生成 Loss Mask ----
        # 预训练阶段：所有有效 token 都参与损失计算
        # 有效位置（前 text_len 个）= 1，padding 位置 = 0
        # 这样模型不会因为 padding token 而产生无意义的梯度
        loss_mask = [1] * text_len + [0] * padding_len

        # ---- 步骤 6：构建因果语言模型的输入输出对 ----
        # Causal LM 的核心思想：用 token[0] 预测 token[1]，用 token[1] 预测 token[2]...
        # 因此：
        #   X = tokens[0 : n-1]    (输入：除了最后一个 token 外的所有 token)
        #   Y = tokens[1 : n  ]    (标签：除了第一个 token 外的所有 token)
        #
        # 示例：原始 tokens = [BOS, 我, 爱, 中国, PAD, PAD]
        #       X           = [BOS, 我, 爱, 中国, PAD]
        #       Y           = [我,  爱, 中国, PAD,  PAD]
        #       loss_mask   = [1,   1,  1,   0,    0  ]  ← 也去掉第一个位置
        #
        # loss_mask 也取 [1:] 保持与 X, Y 对齐
        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)

        # 转换为 PyTorch tensor 并返回
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)


class SFTDataset(Dataset):
    """
    监督微调数据集类 (Supervised Fine-Tuning Dataset)

    用于大语言模型的 SFT 阶段。与预训练不同，SFT 的目标是让模型学会
    遵循指令和对话格式，因此**只对 assistant 的回复部分计算损失**，
    用户输入和 system prompt 部分不计入损失。

    数据格式：
    - 输入文件每行为一个 JSON 对象，符合 chat template 格式
    - 示例：{"messages": [
        {"role": "system", "content": "你是..."},
        {"role": "user", "content": "请问..."},
        {"role": "assistant", "content": "答案是..."}
    ]}

    核心机制 —— Loss Mask 生成：
    SFT 的关键创新在于 loss_mask。它标记了哪些 token 需要计算损失：
    - <|im_start|>assistant\\n 到 <|im_end|>\\n (或 eos) 之间的 token → 损失=1
    - 其他部分（system prompt、user 消息、格式标记）           → 损失=0

    这样模型只会在 assistant 的回复上产生梯度更新，不会因为学到了
    格式标记或用户输入而产生不当的行为偏好。
    """

    def __init__(self, data_path, tokenizer, max_length=512):
        """
        初始化 SFT 数据集

        Args:
            data_path: 数据文件路径，每行一个包含 messages 字段的 JSON 对象
            tokenizer: 分词器实例，需要支持 apply_chat_template 方法和 pad_token_id 属性
            max_length: 序列最大长度，超过的部分将被截断
        """
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 获取 padding token 的 id，若未设置则默认使用 0
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        # ====== 预计算每行的起始字节偏移量（与 PretrainDataset 相同机制）=======
        self._offsets = []
        with open(data_path, 'rb') as f:
            self._offsets.append(0)
            while f.readline():
                self._offsets.append(f.tell())
        self._total_lines = len(self._offsets) - 1

    def __len__(self):
        """返回数据集总样本数"""
        return self._total_lines

    def generate_loss_mask(self, input_ids):
        """
        生成 SFT 损失掩码 —— 只对 assistant 回复部分计算损失

        这是 SFT 训练中最核心的函数。它的目标是识别对话中 assistant 角色的
        发言部分，并只在这些部分标记为 1（参与损失计算），其余部分标记为 0。

        算法原理（滑动窗口匹配）：
        1. 在 token 序列中搜索标记 "<|im_start|>assistant\\n" 的 token 序列
        2. 每找到一个匹配，就从该位置之后开始查找 eos_token
        3. 将 assistant 起始标记之后到 eos_token 之间的所有 token 标记为 1
        4. 跳过已匹配的区域继续搜索（防止重叠匹配）

        处理范围示意：
          <|im_start|>system\\n...<|im_end|>\\n         ← loss = 0
          <|im_start|>user\\n...<|im_end|>\\n           ← loss = 0
          <|im_start|>assistant\\n...<|im_end|>\\n      ← loss = 1 (仅此部分)

        注意：
        - 如果 assistant 回复被 max_length 截断导致没有 eos_token，
          则该段 assistant 回复也不会被标记（因为没有找到结束位置）
        - loss_mask 包含 eos_token 本身（即 end = j，包含位置 j）

        Args:
            input_ids: token 序列 (list 或可索引序列)

        Returns:
            list[int]: 与 input_ids 等长的列表，1=计算损失，0=不计算损失
        """
        # 初始化全零掩码：默认所有位置都不计算损失
        mask = [0] * len(input_ids)

        # 获取 assistant 起始标记的 token 序列
        # 例如 "<|im_start|>assistant\n" → [token_id_1, token_id_2, ...]
        a_sequence = self.tokenizer("<|im_start|>assistant\n")['input_ids']
        a_length = len(a_sequence)          # assistant 起始标记的长度
        n = len(input_ids)                   # 整个序列的长度
        i = 0                                # 滑动窗口的起始位置

        # 滑动窗口搜索：在 input_ids 中查找所有 a_sequence 出现的位置
        while i <= n - a_length:
            # 检查从位置 i 开始是否匹配 assistant 起始标记
            match = True
            for k in range(a_length):
                if input_ids[i + k] != a_sequence[k]:
                    match = False
                    break

            if match:
                # ---- 找到了一个 assistant 起始标记！----
                # 现在需要在它之后查找 eos_token（对话结束标记）
                # 从 assistant 起始标记结束的位置开始向后搜索
                j = None
                for idx in range(i + a_length, n):
                    if input_ids[idx] == self.tokenizer.eos_token_id:
                        j = idx                         # 记录 eos_token 的位置
                        break

                if j is not None:
                    # 找到了 eos_token，标记 [assistant起始后 : eos_token] 为有效区域
                    start = i + a_length                # assistant 回复内容的起点
                    end = j                              # eos_token 的位置（包含）
                    # 将该范围内所有 token 标记为 1（参与损失计算）
                    if start <= end:
                        for pos in range(start, end + 1):
                            if pos < len(mask):
                                mask[pos] = 1

                # 跳过当前匹配的 a_sequence，从其后继续搜索（防止重叠匹配）
                i += a_length
            else:
                # 当前位置不匹配，窗口向右滑动一位
                i += 1

        return mask

    def __getitem__(self, index: int):
        """
        获取第 index 个 SFT 样本

        返回格式与 PretrainDataset 相同：三元组 (X, Y, loss_mask)
        但 loss_mask 的语义不同：只有 assistant 回复部分 = 1。

        Args:
            index: 样本索引，范围 [0, total_lines - 1]

        Returns:
            tuple: (X, Y, loss_mask)，均为 torch.Tensor (int64)
        """
        # ---- 步骤 1：高效读取单行数据 ----
        with open(self.data_path, 'rb') as f:
            f.seek(self._offsets[index])
            line = f.readline().decode('utf-8')

        # ---- 步骤 2：使用 Chat Template 构建文本 ----
        # 解析 JSON，得到 messages 列表
        sample = json.loads(line)

        # apply_chat_template 将 messages 列表转换为格式化文本
        # 例如：{"messages": [{"role": "user", "content": "你好"}]}
        #    → "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"
        # tokenize=False:    先生成文本字符串，稍后统一 tokenize（便于控制截断）
        # add_generation_prompt=False: SFT 阶段用已有回复训练，不需要空白的生成提示
        text = self.tokenizer.apply_chat_template(
            sample,
            tokenize=False,
            add_generation_prompt=False
        )

        # ---- 步骤 3：Tokenize 并截断 ----
        input_id = self.tokenizer(text).data['input_ids'][:self.max_length]
        text_len = len(input_id)

        # ---- 步骤 4：Padding ----
        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len

        # ---- 步骤 5：生成 SFT Loss Mask ----
        # 这是 SFT 与预训练的关键区别：只对 assistant 部分计算损失
        loss_mask = self.generate_loss_mask(input_id)

        # ---- 步骤 6：构建因果语言模型的输入输出对 ----
        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)

        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)
