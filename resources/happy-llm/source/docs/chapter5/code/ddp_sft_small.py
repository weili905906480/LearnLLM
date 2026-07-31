import argparse
import math
import os
import time
from contextlib import nullcontext

import torch
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import SFTDataset
from k_model import ModelConfig, Transformer


def get_lr(step, total_steps, learning_rate, warmup_iters):
    min_lr = learning_rate / 10
    if warmup_iters > 0 and step < warmup_iters:
        return learning_rate * step / warmup_iters
    if step > total_steps:
        return min_lr
    denom = max(total_steps - warmup_iters, 1)
    decay_ratio = (step - warmup_iters) / denom
    decay_ratio = min(max(decay_ratio, 0), 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description="Small Tiny-K SFT")
    parser.add_argument("--out_dir", type=str, default="sft_model_small")
    parser.add_argument("--data_path", type=str, default="./BelleGroup_sft_learnllm.jsonl")
    parser.add_argument("--pretrain_ckpt", type=str, default="./base_model_small/pretrain_128_2_6144.pth")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_kv_heads", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(42)

    tokenizer = AutoTokenizer.from_pretrained("./tokenizer_k/")
    lm_config = ModelConfig(
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        max_seq_len=args.max_seq_len,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
    )
    model = Transformer(lm_config)
    if args.pretrain_ckpt:
        state_dict = torch.load(args.pretrain_ckpt, map_location=args.device)
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrain checkpoint from {args.pretrain_ckpt}")
    model = model.to(args.device)
    print(f"Small LLM parameters: {count_parameters(model) / 1e6:.3f} M")

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=("cuda" in args.device),
        drop_last=False,
        shuffle=True,
        num_workers=args.num_workers,
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=("cuda" in args.device and args.dtype in ["float16", "bfloat16"]))
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    ctx = nullcontext() if "cuda" not in args.device else torch.amp.autocast(device_type="cuda", dtype=ptdtype)

    iter_per_epoch = len(train_loader)
    total_steps = args.epochs * iter_per_epoch
    global_step = 0
    start_time = time.time()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for step, (X, Y, loss_mask) in enumerate(train_loader):
            X = X.to(args.device)
            Y = Y.to(args.device)
            loss_mask = loss_mask.to(args.device).view(-1)

            lr = get_lr(global_step, total_steps, args.learning_rate, args.warmup_iters)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            with ctx:
                out = model(X, Y)
                token_loss = out.last_loss / args.accumulation_steps
                loss = torch.sum(token_loss * loss_mask) / loss_mask.sum()

            scaler.scale(loss).backward()
            if (step + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if global_step % args.log_interval == 0:
                elapsed = time.time() - start_time
                print(
                    f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iter_per_epoch}) "
                    f"loss:{loss.item() * args.accumulation_steps:.3f} lr:{lr:.7f} elapsed:{elapsed / 60:.1f}min"
                )

            if (global_step + 1) % args.save_interval == 0:
                ckp = f"{args.out_dir}/sft_dim{lm_config.dim}_layers{lm_config.n_layers}_vocab_size{lm_config.vocab_size}.pth"
                torch.save(model.state_dict(), ckp)

            global_step += 1

    ckp = f"{args.out_dir}/sft_dim{lm_config.dim}_layers{lm_config.n_layers}_vocab_size{lm_config.vocab_size}.pth"
    torch.save(model.state_dict(), ckp)
    print(f"Saved SFT checkpoint to {ckp}")


if __name__ == "__main__":
    main()
