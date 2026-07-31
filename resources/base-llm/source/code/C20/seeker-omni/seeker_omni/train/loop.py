import itertools
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import ExperimentConfig
from ..dataset.memmap import MemmapDataset
from ..model.lm import SeekerOmniLM
from .checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from .freezing import apply_stage_freeze
from .lr import cosine_lr


def _maybe_tb_writer(cfg: ExperimentConfig):
    if not bool(getattr(cfg.train, "tb_enable", True)):
        return None

    from torch.utils.tensorboard import SummaryWriter

    tb_dir = getattr(cfg.train, "tb_dir", None)
    if tb_dir is None:
        out_dir = Path(cfg.train.out_dir)
        parts = out_dir.parts
        rel = Path(*parts[1:]) if (len(parts) >= 2 and parts[0].lower() == "checkpoints") else out_dir
        tb_dir = Path("outputs") / "tb" / rel
    tb_every = int(getattr(cfg.train, "tb_every", cfg.train.log_every))
    return SummaryWriter(log_dir=str(tb_dir)), tb_every


def _pick_device(device: str) -> torch.device:
    if device == 'cpu':
        return torch.device('cpu')
    if device == 'cuda':
        return torch.device('cuda')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _pick_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if dtype == 'fp32':
        return torch.float32
    if dtype == 'fp16':
        return torch.float16
    if dtype == 'bf16':
        if device.type == 'cuda' and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if device.type == 'cpu':
            return torch.bfloat16
        return torch.float16
    raise ValueError(f'unknown dtype: {dtype}')


def _normalize_mix_weights(num: int, weights: tuple[float, ...] | None) -> list[float]:
    if weights is None:
        return [1.0 / float(num)] * int(num)

    if len(weights) != int(num):
        raise ValueError(f'mix_weights length mismatch: got={len(weights)} expected={num}')

    w = [float(x) for x in weights]
    if any((not (x >= 0.0)) for x in w):
        raise ValueError('mix_weights must be non-negative')

    s = float(sum(w))
    if not (s > 0.0):
        raise ValueError('mix_weights sum must be > 0')

    return [x / s for x in w]


def _prepare_batch(batch: dict, *, device: torch.device):
    input_ids = batch['input_ids'].to(device=device, dtype=torch.long)
    labels = batch['labels'].to(device=device, dtype=torch.long)
    attention_mask_cpu = batch.get('attention_mask')
    if attention_mask_cpu is None:
        attention_mask = None
    else:
        # 如果 attention_mask 全为 1，表示完全 packed、无 padding；此时不传给 SDPA，
        # 以便使用 flash attention 并减少显存占用。
        if bool((attention_mask_cpu.min() == 1).item()):
            attention_mask = None
        else:
            attention_mask = attention_mask_cpu.to(device=device, dtype=torch.float32)

    image_feats = batch.get('image_feats')
    if image_feats is not None:
        image_feats = image_feats.to(
            device=device,
            dtype=(torch.float32 if device.type == 'cpu' else torch.float16),
        )
    return input_ids, labels, attention_mask, image_feats


def _corrupt_answer_tokens_for_mm(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    unk_id: int,
    n_special: int,
) -> torch.Tensor:
    """在 *input* 中扰动监督答案 token，以降低教师强制。

    这样前几个助手 token 会更多依赖提示词 + 模态特征（I/A），
    而不是上下文里已经包含了金标答案前缀。
    """
    mask = labels != -100
    if not bool(mask.any()):
        return input_ids

    # 固定策略：答案开头先扰动一小段，再对答案区间做随机扰动。
    prefix_k = 16
    drop_p = 0.20

    out = input_ids.clone()

    has = mask.sum(dim=1) > 0
    first = torch.argmax(mask.to(torch.int64), dim=1)
    for b in range(int(out.shape[0])):
        if not bool(has[b]):
            continue
        s = int(first[b].item())
        e = min(s + int(prefix_k), int(out.shape[1]))
        out[b, s:e] = int(unk_id)

    if drop_p > 0.0:
        drop_mask = mask & (out >= int(n_special))
        if bool(drop_mask.any()):
            rand = torch.rand(out.shape, device=out.device)
            drop_mask = drop_mask & (rand < float(drop_p))
            if bool(drop_mask.any()):
                out[drop_mask] = int(unk_id)

    return out


def _adamw_param_groups(model: SeekerOmniLM, *, weight_decay: float) -> list[dict]:
    """为 AdamW 构建参数组，并对权重衰减做选择性过滤。

    说明：
    - 不对 norm、词嵌入、模态适配器、门控 做权重衰减。
    - 门控参数初始接近 0，强 L2 正则下可能比较脆弱。
    """
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        lname = name.lower()
        is_bias = lname.endswith(".bias") or lname.endswith("bias")
        is_norm = "norm" in lname
        is_gate = lname.endswith("img_gate")
        is_embed = lname.endswith("special_embed.weight") or lname.endswith("base_embed.weight")
        is_mm_adapter = lname.endswith("img_proj.weight")

        if is_bias or is_norm or is_gate or is_embed or is_mm_adapter:
            no_decay.append(p)
        else:
            decay.append(p)

    groups: list[dict] = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(weight_decay)})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def train(cfg: ExperimentConfig) -> None:
    device = _pick_device(cfg.train.device)
    dtype = _pick_dtype(cfg.train.dtype, device)

    mix_dirs = cfg.data.mix_train_dirs
    mix_enabled = bool(mix_dirs)

    if mix_enabled and cfg.train.max_steps is None:
        raise ValueError('data.mix_train_dirs requires train.max_steps (mixed training is step-based)')

    if mix_enabled:
        mix_dirs = tuple(mix_dirs or ())
        probs = _normalize_mix_weights(len(mix_dirs), cfg.data.mix_weights)

        datasets = [MemmapDataset(p) for p in mix_dirs]
        for p, ds in zip(mix_dirs, datasets):
            if int(ds.meta.seq_len) != int(cfg.model.max_seq_len):
                raise ValueError(f'seq_len mismatch for {p}: data={ds.meta.seq_len} model={cfg.model.max_seq_len}')
            if int(ds.meta.vocab_size) != int(cfg.model.vocab_size):
                raise ValueError(f'vocab_size mismatch for {p}: data={ds.meta.vocab_size} model={cfg.model.vocab_size}')

        dls = [DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=4) for ds in datasets]
        iters = [iter(dl) for dl in dls]
        rng = random.Random(int(cfg.seed))

        def next_mixed_batch() -> tuple[dict, int]:
            idx = rng.choices(range(len(dls)), weights=probs, k=1)[0]
            try:
                batch = next(iters[idx])
            except StopIteration:
                iters[idx] = iter(dls[idx])
                batch = next(iters[idx])
            return batch, int(idx)

        total_steps = int(cfg.train.max_steps)
    else:
        ds = MemmapDataset(cfg.data.train_dir)
        if int(ds.meta.seq_len) != int(cfg.model.max_seq_len):
            raise ValueError(f'seq_len mismatch: data={ds.meta.seq_len} model={cfg.model.max_seq_len}')
        if int(ds.meta.vocab_size) != int(cfg.model.vocab_size):
            raise ValueError(f'vocab_size mismatch: data={ds.meta.vocab_size} model={cfg.model.vocab_size}')

        dl = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=4)
        total_steps = cfg.train.epochs * len(dl)
        if cfg.train.max_steps is not None:
            total_steps = int(cfg.train.max_steps)

    model = SeekerOmniLM(cfg.model).to(device)
    apply_stage_freeze(model, cfg.train)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(_adamw_param_groups(model, weight_decay=cfg.train.weight_decay), lr=cfg.train.lr)

    # stage 切换时的 init_from（仅模型权重）。
    if cfg.train.init_from is not None:
        load_checkpoint(cfg.train.init_from, model=model, optimizer=None)

    # 同一 stage/out_dir 内的断点续训。
    start_step = 0
    ckpt = latest_checkpoint(cfg.train.out_dir)
    if ckpt is not None:
        start_step = load_checkpoint(ckpt, model=model, optimizer=opt)

    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == 'cuda' and dtype == torch.float16))

    model.train()
    step = start_step
    t0 = time.time()

    tb = _maybe_tb_writer(cfg)
    tb_writer = tb[0] if tb is not None else None
    tb_every = int(tb[1]) if tb is not None else 0

    if mix_enabled:
        pbar = tqdm(total=int(total_steps), initial=int(step), desc='train(mix)')
        while step < int(total_steps):
            step += 1

            lr = cosine_lr(step, base_lr=cfg.train.lr, total_steps=int(total_steps), warmup_steps=cfg.train.warmup_steps)
            for g in opt.param_groups:
                g['lr'] = lr

            batch, src = next_mixed_batch()
            input_ids, labels, attention_mask, image_feats = _prepare_batch(batch, device=device)

            if image_feats is not None:
                input_ids = _corrupt_answer_tokens_for_mm(
                    input_ids,
                    labels,
                    unk_id=int(model.special.unk),
                    n_special=int(model.n_special),
                )

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type != 'cpu' or dtype != torch.float32)):
                out = model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    image_feats=image_feats,
                )
                loss = out.loss / cfg.train.grad_accum

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % cfg.train.grad_accum == 0:
                if scaler.is_enabled():
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)

                if scaler.is_enabled():
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)

            if step % cfg.train.log_every == 0:
                elapsed = time.time() - t0
                if device.type == 'cuda':
                    alloc = int(torch.cuda.memory_allocated(device) / (1024**2))
                    reserv = int(torch.cuda.memory_reserved(device) / (1024**2))
                    pbar.set_postfix(
                        loss=float(loss.detach().cpu()) * cfg.train.grad_accum,
                        lr=lr,
                        sec=int(elapsed),
                        mem=f'{alloc}/{reserv}MiB',
                        src=src,
                    )
                else:
                    pbar.set_postfix(loss=float(loss.detach().cpu()) * cfg.train.grad_accum, lr=lr, sec=int(elapsed), src=src)

            if tb_writer is not None and tb_every > 0 and (step % tb_every == 0):
                with torch.no_grad():
                    tb_writer.add_scalar("train/loss", float(out.loss.detach().float().cpu()), global_step=int(step))
                    tb_writer.add_scalar("train/ppl", float(torch.exp(out.loss.detach().float()).cpu()), global_step=int(step))
                    tb_writer.add_scalar("train/lr", float(lr), global_step=int(step))
                    tb_writer.add_scalar("train/src", int(src), global_step=int(step))
                    tb_writer.add_scalar(
                        "train/img_gate_tanh_abs_mean",
                        float(torch.tanh(model.img_gate).abs().mean().detach().float().cpu()),
                        global_step=int(step),
                    )
                    if device.type == "cuda":
                        tb_writer.add_scalar(
                            "train/cuda_mem_alloc_mib",
                            float(torch.cuda.memory_allocated(device) / (1024**2)),
                            global_step=int(step),
                        )
                        tb_writer.add_scalar(
                            "train/cuda_mem_reserved_mib",
                            float(torch.cuda.memory_reserved(device) / (1024**2)),
                            global_step=int(step),
                        )

            if step % cfg.train.save_every == 0:
                save_checkpoint(
                    cfg.train.out_dir,
                    model=model,
                    optimizer=opt,
                    step=step,
                    cfg=asdict(cfg),
                    keep_last=cfg.train.keep_last,
                )

            pbar.update(1)

        pbar.close()
        save_checkpoint(
            cfg.train.out_dir,
            model=model,
            optimizer=opt,
            step=step,
            cfg=asdict(cfg),
            keep_last=cfg.train.keep_last,
        )
        if tb_writer is not None:
            tb_writer.close()
        return

    # 单数据集训练。
    if cfg.train.max_steps is not None:
        # 基于 step：进度显示为 step/max_steps（避免提前停止时 epoch 进度产生误解）。
        dl_iter = iter(dl)
        pbar = tqdm(total=int(total_steps), initial=int(step), desc='train')

        while step < int(total_steps):
            try:
                batch = next(dl_iter)
            except StopIteration:
                dl_iter = iter(dl)
                batch = next(dl_iter)

            step += 1
            lr = cosine_lr(step, base_lr=cfg.train.lr, total_steps=int(total_steps), warmup_steps=cfg.train.warmup_steps)
            for g in opt.param_groups:
                g['lr'] = lr

            input_ids, labels, attention_mask, image_feats = _prepare_batch(batch, device=device)

            # 多模态样本降低教师强制，鼓励使用模态特征。
            if image_feats is not None:
                input_ids = _corrupt_answer_tokens_for_mm(
                    input_ids,
                    labels,
                    unk_id=int(model.special.unk),
                    n_special=int(model.n_special),
                )

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type != 'cpu' or dtype != torch.float32)):
                out = model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    image_feats=image_feats,
                )
                loss = out.loss / cfg.train.grad_accum

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % cfg.train.grad_accum == 0:
                if scaler.is_enabled():
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)

                if scaler.is_enabled():
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)

            if step % cfg.train.log_every == 0:
                elapsed = time.time() - t0
                img_gate = float(torch.tanh(model.img_gate).abs().mean().detach().cpu())
                if device.type == 'cuda':
                    alloc = int(torch.cuda.memory_allocated(device) / (1024**2))
                    reserv = int(torch.cuda.memory_reserved(device) / (1024**2))
                    pbar.set_postfix(
                        loss=float(loss.detach().cpu()) * cfg.train.grad_accum,
                        lr=lr,
                        sec=int(elapsed),
                        mem=f'{alloc}/{reserv}MiB',
                        img_gate=img_gate,
                    )
                else:
                    pbar.set_postfix(
                        loss=float(loss.detach().cpu()) * cfg.train.grad_accum,
                        lr=lr,
                        sec=int(elapsed),
                        img_gate=img_gate,
                    )

            if tb_writer is not None and tb_every > 0 and (step % tb_every == 0):
                with torch.no_grad():
                    tb_writer.add_scalar("train/loss", float(out.loss.detach().float().cpu()), global_step=int(step))
                    tb_writer.add_scalar("train/ppl", float(torch.exp(out.loss.detach().float()).cpu()), global_step=int(step))
                    tb_writer.add_scalar("train/lr", float(lr), global_step=int(step))
                    tb_writer.add_scalar(
                        "train/img_gate_tanh_abs_mean",
                        float(torch.tanh(model.img_gate).abs().mean().detach().float().cpu()),
                        global_step=int(step),
                    )
                    if device.type == "cuda":
                        tb_writer.add_scalar(
                            "train/cuda_mem_alloc_mib",
                            float(torch.cuda.memory_allocated(device) / (1024**2)),
                            global_step=int(step),
                        )
                        tb_writer.add_scalar(
                            "train/cuda_mem_reserved_mib",
                            float(torch.cuda.memory_reserved(device) / (1024**2)),
                            global_step=int(step),
                        )

            if step % cfg.train.save_every == 0:
                save_checkpoint(
                    cfg.train.out_dir,
                    model=model,
                    optimizer=opt,
                    step=step,
                    cfg=asdict(cfg),
                    keep_last=cfg.train.keep_last,
                )

            pbar.update(1)

        pbar.close()
        save_checkpoint(
            cfg.train.out_dir,
            model=model,
            optimizer=opt,
            step=step,
            cfg=asdict(cfg),
            keep_last=cfg.train.keep_last,
        )
        if tb_writer is not None:
            tb_writer.close()
        return

    # 基于 epoch。
    for epoch in range(cfg.train.epochs):
        pbar = tqdm(dl, desc=f'epoch {epoch+1}')
        for batch in pbar:
            step += 1
            lr = cosine_lr(step, base_lr=cfg.train.lr, total_steps=int(total_steps), warmup_steps=cfg.train.warmup_steps)
            for g in opt.param_groups:
                g['lr'] = lr

            input_ids, labels, attention_mask, image_feats = _prepare_batch(batch, device=device)

            # 多模态样本降低教师强制，鼓励使用模态特征。
            if image_feats is not None:
                input_ids = _corrupt_answer_tokens_for_mm(
                    input_ids,
                    labels,
                    unk_id=int(model.special.unk),
                    n_special=int(model.n_special),
                )

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type != 'cpu' or dtype != torch.float32)):
                out = model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    image_feats=image_feats,
                )
                loss = out.loss / cfg.train.grad_accum

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % cfg.train.grad_accum == 0:
                if scaler.is_enabled():
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)

                if scaler.is_enabled():
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)

            if step % cfg.train.log_every == 0:
                elapsed = time.time() - t0
                img_gate = float(torch.tanh(model.img_gate).abs().mean().detach().cpu())
                if device.type == 'cuda':
                    alloc = int(torch.cuda.memory_allocated(device) / (1024**2))
                    reserv = int(torch.cuda.memory_reserved(device) / (1024**2))
                    pbar.set_postfix(
                        loss=float(loss.detach().cpu()) * cfg.train.grad_accum,
                        lr=lr,
                        sec=int(elapsed),
                        mem=f'{alloc}/{reserv}MiB',
                        img_gate=img_gate,
                    )
                else:
                    pbar.set_postfix(
                        loss=float(loss.detach().cpu()) * cfg.train.grad_accum,
                        lr=lr,
                        sec=int(elapsed),
                        img_gate=img_gate,
                    )

            if tb_writer is not None and tb_every > 0 and (step % tb_every == 0):
                with torch.no_grad():
                    tb_writer.add_scalar("train/loss", float(out.loss.detach().float().cpu()), global_step=int(step))
                    tb_writer.add_scalar("train/ppl", float(torch.exp(out.loss.detach().float()).cpu()), global_step=int(step))
                    tb_writer.add_scalar("train/lr", float(lr), global_step=int(step))
                    tb_writer.add_scalar("train/epoch", float(epoch + 1), global_step=int(step))
                    tb_writer.add_scalar(
                        "train/img_gate_tanh_abs_mean",
                        float(torch.tanh(model.img_gate).abs().mean().detach().float().cpu()),
                        global_step=int(step),
                    )
                    if device.type == "cuda":
                        tb_writer.add_scalar(
                            "train/cuda_mem_alloc_mib",
                            float(torch.cuda.memory_allocated(device) / (1024**2)),
                            global_step=int(step),
                        )
                        tb_writer.add_scalar(
                            "train/cuda_mem_reserved_mib",
                            float(torch.cuda.memory_reserved(device) / (1024**2)),
                            global_step=int(step),
                        )

            if step % cfg.train.save_every == 0:
                save_checkpoint(
                    cfg.train.out_dir,
                    model=model,
                    optimizer=opt,
                    step=step,
                    cfg=asdict(cfg),
                    keep_last=cfg.train.keep_last,
                )

    save_checkpoint(
        cfg.train.out_dir,
        model=model,
        optimizer=opt,
        step=step,
        cfg=asdict(cfg),
        keep_last=cfg.train.keep_last,
    )
    if tb_writer is not None:
        tb_writer.close()
