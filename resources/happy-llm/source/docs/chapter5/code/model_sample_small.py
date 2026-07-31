import argparse
from contextlib import nullcontext

import torch
from transformers import AutoTokenizer

from k_model import ModelConfig, Transformer


class SmallTextGenerator:
    def __init__(
        self,
        checkpoint='./sft_model_small/sft_dim128_layers2_vocab_size6144.pth',
        tokenizer_model_path='./tokenizer_k/',
        device=None,
        dtype='float32',
        dim=128,
        n_layers=2,
        n_heads=8,
        n_kv_heads=4,
        max_seq_len=256,
    ):
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device_type = 'cuda' if 'cuda' in self.device else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path)
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=ptdtype)
        config = ModelConfig(
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            max_seq_len=max_seq_len,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0,
        )
        self.model = Transformer(config)
        state_dict = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval().to(self.device)
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Model has {num_params / 1e6:.3f} M parameters.')

    def chat_template(self, prompt):
        messages = [
            {'role': 'system', 'content': '你是一个耐心、准确的LLM学习助手，回答要简洁清楚。'},
            {'role': 'user', 'content': prompt},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def sample(self, prompt, max_new_tokens=80, temperature=0.4, top_k=40):
        text = self.chat_template(prompt)
        input_ids = self.tokenizer(text).data['input_ids']
        x = torch.tensor(input_ids, dtype=torch.long, device=self.device)[None, ...]
        with torch.no_grad():
            with self.ctx:
                y = self.model.generate(
                    x,
                    stop_id=self.tokenizer.eos_token_id,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                )
        return self.tokenizer.decode(y[0].tolist(), skip_special_tokens=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Small/medium Tiny-K inference')
    parser.add_argument('--checkpoint', default='./sft_model_small/sft_dim128_layers2_vocab_size6144.pth')
    parser.add_argument('--prompt', default='SFT是什么意思？')
    parser.add_argument('--max_new_tokens', type=int, default=80)
    parser.add_argument('--temperature', type=float, default=0.4)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--tokenizer_model_path', default='./tokenizer_k/')
    parser.add_argument('--device', default=None)
    parser.add_argument('--dtype', default='float32', choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_kv_heads', type=int, default=4)
    parser.add_argument('--max_seq_len', type=int, default=256)
    args = parser.parse_args()
    generator = SmallTextGenerator(
        checkpoint=args.checkpoint,
        tokenizer_model_path=args.tokenizer_model_path,
        device=args.device,
        dtype=args.dtype,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        max_seq_len=args.max_seq_len,
    )
    print(generator.sample(args.prompt, args.max_new_tokens, args.temperature, args.top_k))
