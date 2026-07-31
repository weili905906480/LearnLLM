import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .distill import mse_distill
from .vision import default_tb_dir, freeze_vision_all_but_last_n, load_rgb, pool_tokens_torch

from seeker_omni.config import ExperimentConfig
from seeker_omni.dataset.schema import iter_jsonl
from seeker_omni.model.lm import SeekerOmniLM
from seeker_omni.model.resampler import PerceiverResampler
from seeker_omni.paths import MM_TRAIN_JSONL, TOKENIZER_DIR
from seeker_omni.train.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from seeker_omni.train.loop import _corrupt_answer_tokens_for_mm
from seeker_omni.train.lr import cosine_lr


class JsonlImageOnlySFTDataset(Dataset):
    def __init__(
        self,
        *,
        jsonl_path: str | Path,
        text_tokenizer: Tokenizer,
        max_seq_len: int,
        image_tokens: int,
        id_contains: str | None = None,
        max_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.tok = text_tokenizer
        self.max_seq_len = int(max_seq_len)
        self.image_tokens = int(image_tokens)

        # 只支持 minimind2_chatml：<|endoftext|>/<|im_start|>/<|im_end|> + <img_*>。
        pad_id = self.tok.token_to_id("<|endoftext|>")
        if pad_id is None:
            raise SystemExit("tokenizer missing <|endoftext|> (pad)")
        self.pad_id = int(pad_id)

        bos_id = self.tok.token_to_id("<|im_start|>")
        if bos_id is None:
            raise SystemExit("tokenizer missing <|im_start|> (bos)")
        self.bos_id = int(bos_id)

        eos_id = self.tok.token_to_id("<|im_end|>")
        if eos_id is None:
            raise SystemExit("tokenizer missing <|im_end|> (eos)")
        self.eos_id = int(eos_id)

        img_bos_id = self.tok.token_to_id("<img_bos>")
        img_id = self.tok.token_to_id("<img>")
        img_eos_id = self.tok.token_to_id("<img_eos>")
        missing = [
            name
            for name, tid in (("img_bos", img_bos_id), ("img", img_id), ("img_eos", img_eos_id))
            if tid is None
        ]
        if missing:
            raise SystemExit(f"tokenizer missing image special tokens: {missing}")
        self.img_bos_id = int(img_bos_id)
        self.img_id = int(img_id)
        self.img_eos_id = int(img_eos_id)

        items = []
        for s in iter_jsonl(jsonl_path):
            if not s.image:
                continue
            if id_contains and id_contains not in str(s.id):
                continue
            if not s.answer.strip():
                continue
            items.append(
                {
                    "id": s.id,
                    "system": s.system or "你是一个只用中文回答的助手。",
                    "prompt": s.prompt.strip() or "请用中文描述图片。",
                    "answer": s.answer.strip(),
                    "image": str(s.image),
                }
            )

        if not items:
            raise SystemExit("no image-only samples found in jsonl (check filters / path)")

        if max_samples is not None and int(max_samples) > 0 and len(items) > int(max_samples):
            rnd = random.Random(int(seed))
            items = rnd.sample(items, k=int(max_samples))

        self.items = items

    def __len__(self) -> int:
        return int(len(self.items))

    def _enc(self, s: str) -> list[int]:
        return self.tok.encode(s).ids

    def __getitem__(self, idx: int) -> dict:
        it = self.items[int(idx)]
        system = str(it["system"])
        prompt = str(it["prompt"])
        answer = str(it["answer"])

        nl = self._enc("\n")
        tokens: list[int] = [self.bos_id] + self._enc("system\n") + self._enc(system) + [self.eos_id] + nl
        tokens += [self.bos_id] + self._enc("user\n") + self._enc(prompt) + nl

        tokens += [self.img_bos_id] + [self.img_id] * int(self.image_tokens) + [self.img_eos_id]

        nl = self._enc("\n")
        tokens += [self.eos_id] + nl
        tokens += [self.bos_id] + self._enc("assistant\n")
        answer_start = len(tokens)
        tokens += self._enc(answer) + [self.eos_id] + nl

        if len(tokens) > int(self.max_seq_len):
            tokens = tokens[: int(self.max_seq_len)]

        input_ids = torch.full((int(self.max_seq_len),), int(self.pad_id), dtype=torch.long)
        input_ids[: len(tokens)] = torch.tensor(tokens, dtype=torch.long)

        attention_mask = torch.zeros((int(self.max_seq_len),), dtype=torch.float32)
        attention_mask[: len(tokens)] = 1.0

        labels = torch.full((int(self.max_seq_len),), -100, dtype=torch.long)
        ans_start = min(int(answer_start), int(len(tokens)))
        labels[ans_start : len(tokens)] = input_ids[ans_start : len(tokens)]
        labels[input_ids == int(self.pad_id)] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "image_path": str(it["image"]),
            "id": str(it["id"]),
        }


class JsonlTextOnlySFTDataset(Dataset):
    def __init__(
        self,
        *,
        jsonl_path: str | Path,
        text_tokenizer: Tokenizer,
        max_seq_len: int,
        image_tokens: int,
        id_contains: str | None = None,
        max_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.tok = text_tokenizer
        self.max_seq_len = int(max_seq_len)
        self.image_tokens = int(image_tokens)

        pad_id = self.tok.token_to_id("<|endoftext|>")
        if pad_id is None:
            raise SystemExit("tokenizer missing <|endoftext|> (pad)")
        self.pad_id = int(pad_id)

        bos_id = self.tok.token_to_id("<|im_start|>")
        if bos_id is None:
            raise SystemExit("tokenizer missing <|im_start|> (bos)")
        self.bos_id = int(bos_id)

        eos_id = self.tok.token_to_id("<|im_end|>")
        if eos_id is None:
            raise SystemExit("tokenizer missing <|im_end|> (eos)")
        self.eos_id = int(eos_id)

        img_bos_id = self.tok.token_to_id("<img_bos>")
        img_id = self.tok.token_to_id("<img>")
        img_eos_id = self.tok.token_to_id("<img_eos>")
        missing = [
            name
            for name, tid in (("img_bos", img_bos_id), ("img", img_id), ("img_eos", img_eos_id))
            if tid is None
        ]
        if missing:
            raise SystemExit(f"tokenizer missing image special tokens: {missing}")
        self.img_bos_id = int(img_bos_id)
        self.img_id = int(img_id)
        self.img_eos_id = int(img_eos_id)

        items = []
        for s in iter_jsonl(jsonl_path):
            if bool(s.image):
                continue
            if id_contains and id_contains not in str(s.id):
                continue
            if not s.prompt.strip():
                continue
            if not s.answer.strip():
                continue
            items.append(
                {
                    "id": s.id,
                    "system": s.system or "你是一个只用中文回答的助手。",
                    "prompt": s.prompt.strip(),
                    "answer": s.answer.strip(),
                }
            )

        if not items:
            raise SystemExit("no text-only samples found in jsonl (check filters / path)")

        if max_samples is not None and int(max_samples) > 0 and len(items) > int(max_samples):
            rnd = random.Random(int(seed))
            items = rnd.sample(items, k=int(max_samples))

        self.items = items

    def __len__(self) -> int:
        return int(len(self.items))

    def _enc(self, s: str) -> list[int]:
        return self.tok.encode(s).ids

    def __getitem__(self, idx: int) -> dict:
        it = self.items[int(idx)]
        system = str(it["system"])
        prompt = str(it["prompt"])
        answer = str(it["answer"])

        nl = self._enc("\n")
        tokens: list[int] = [self.bos_id] + self._enc("system\n") + self._enc(system) + [self.eos_id] + nl
        tokens += [self.bos_id] + self._enc("user\n") + self._enc(prompt) + nl

        img_seg_start = len(tokens)
        tokens += [self.img_bos_id] + [self.img_id] * int(self.image_tokens) + [self.img_eos_id]
        img_seg_end = len(tokens)

        nl = self._enc("\n")
        tokens += [self.eos_id] + nl
        tokens += [self.bos_id] + self._enc("assistant\n")
        answer_start = len(tokens)
        tokens += self._enc(answer) + [self.eos_id] + nl

        if len(tokens) > int(self.max_seq_len):
            tokens = tokens[: int(self.max_seq_len)]

        input_ids = torch.full((int(self.max_seq_len),), int(self.pad_id), dtype=torch.long)
        input_ids[: len(tokens)] = torch.tensor(tokens, dtype=torch.long)

        attention_mask = torch.zeros((int(self.max_seq_len),), dtype=torch.float32)
        attention_mask[: len(tokens)] = 1.0

        # 纯文本：把图像占位符 mask 掉。
        s = min(int(img_seg_start), len(tokens))
        e = min(int(img_seg_end), len(tokens))
        input_ids[s:e] = int(self.pad_id)
        attention_mask[s:e] = 0.0

        labels = torch.full((int(self.max_seq_len),), -100, dtype=torch.long)
        ans_start = min(int(answer_start), int(len(tokens)))
        labels[ans_start : len(tokens)] = input_ids[ans_start : len(tokens)]
        labels[input_ids == int(self.pad_id)] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "image_path": "",
            "id": str(it["id"]),
        }


def _freeze_llm_keep_mm_adapters(model: SeekerOmniLM) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for p in model.img_proj.parameters():
        p.requires_grad = True
    model.img_gate.requires_grad = True


def _split_model_param_groups(model: SeekerOmniLM) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    bridge: list[torch.nn.Parameter] = []
    llm: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("img_proj.") or name in ("img_gate",):
            bridge.append(p)
        else:
            llm.append(p)
    return bridge, llm


def _resolve_start_ckpt(exp_cfg: ExperimentConfig, cfg: dict) -> Path:
    ckpt_raw = cfg.get("ckpt")
    if ckpt_raw:
        ckpt = Path(str(ckpt_raw))
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        return ckpt

    ckpt = latest_checkpoint(exp_cfg.train.out_dir)
    if ckpt is None or not ckpt.exists():
        raise RuntimeError(f"no checkpoint found in: {exp_cfg.train.out_dir}")
    return ckpt


def _args_from_yaml(e2e_cfg: dict, ckpt: Path) -> SimpleNamespace:
    train_cfg = e2e_cfg.get("train", {})
    vision_cfg = e2e_cfg.get("vision", {})
    tb_cfg = e2e_cfg.get("tb", {})

    # 教学默认：稳定的 E2E 视觉微调（avgpool；冻结 LLM）。
    return SimpleNamespace(
        config=str(e2e_cfg["config"]),
        ckpt=str(ckpt),
        text_tokenizer=str(e2e_cfg.get("text_tokenizer", str(TOKENIZER_DIR))),
        jsonl=str(e2e_cfg.get("jsonl", str(MM_TRAIN_JSONL))),
        vision_model=str(e2e_cfg.get("vision_model", "google/siglip2-base-patch16-224")),
        out_dir=str(e2e_cfg.get("out_dir", "checkpoints/e2e_imgcap_stable")),
        vision_out_dir=str(e2e_cfg.get("vision_out_dir", "checkpoints/e2e_imgcap_stable/vision")),
        device=str(train_cfg.get("device", "cuda")),
        dtype=str(train_cfg.get("dtype", "fp16")),
        seed=int(e2e_cfg.get("seed", 42)),
        max_samples=int(e2e_cfg.get("max_samples", 8000)),
        id_contains=(str(e2e_cfg["id_contains"]) if e2e_cfg.get("id_contains") else None),
        batch_size=int(train_cfg.get("batch_size", 1)),
        grad_accum=int(train_cfg.get("grad_accum", 8)),
        max_steps=int(train_cfg.get("max_steps", 3000)),
        lr=float(train_cfg.get("lr", 1.0e-5)),
        lr_bridge=float(train_cfg.get("lr_bridge", 2.0e-5)),
        lr_resampler=0.0,
        lr_vision=float(train_cfg.get("lr_vision", 1.0e-6)),
        lr_llm=0.0,
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        warmup_steps=int(train_cfg.get("warmup_steps", 200)),
        grad_clip=float(train_cfg.get("grad_clip", 1.0)),
        log_every=int(train_cfg.get("log_every", 50)),
        save_every=int(train_cfg.get("save_every", 500)),
        keep_last=int(train_cfg.get("keep_last", 2)),
        resume_optimizer=int(train_cfg.get("resume_optimizer", 1)),
        unfreeze_llm=False,
        vision_train_last_n=int(vision_cfg.get("vision_train_last_n", 2)),
        freeze_vision=False,
        image_embed="avgpool",
        resampler_layers=2,
        resampler_heads=8,
        resampler_ff_mult=4,
        mm_corrupt=0,
        vision_distill_weight=float(vision_cfg.get("distill_weight", 0.1)),
        text_jsonl="",
        text_mix_ratio=0.0,
        text_max_samples=0,
        tb_enable=(1 if bool(tb_cfg.get("enable", True)) else 0),
        tb_dir=str(tb_cfg.get("dir", "")),
        tb_every=int(tb_cfg.get("every", 0)),
    )


def run(args: SimpleNamespace) -> int:
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    cfg = ExperimentConfig.load(args.config)
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else args.device)

    if args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    tok = Tokenizer.from_file(str(Path(args.text_tokenizer) / "tokenizer.json"))
    if tok.get_vocab_size() != int(cfg.model.vocab_size):
        raise SystemExit(f"text vocab mismatch: tokenizer={tok.get_vocab_size()} cfg={cfg.model.vocab_size}")

    ds = JsonlImageOnlySFTDataset(
        jsonl_path=args.jsonl,
        text_tokenizer=tok,
        max_seq_len=int(cfg.model.max_seq_len),
        image_tokens=int(cfg.model.image_tokens),
        id_contains=(str(args.id_contains) if args.id_contains else None),
        max_samples=(int(args.max_samples) if int(args.max_samples) > 0 else None),
        seed=int(args.seed),
    )
    dl = DataLoader(ds, batch_size=int(args.batch_size), shuffle=True, num_workers=0)
    text_dl = None
    if str(args.text_jsonl).strip():
        if not bool(args.unfreeze_llm) and float(args.text_mix_ratio) > 0.0:
            print("note: --text_mix_ratio ignored because --unfreeze_llm is not set (no trainable text-only params).")
        else:
            text_ds = JsonlTextOnlySFTDataset(
                jsonl_path=str(args.text_jsonl),
                text_tokenizer=tok,
                max_seq_len=int(cfg.model.max_seq_len),
                image_tokens=int(cfg.model.image_tokens),
                id_contains=None,
                max_samples=(int(args.text_max_samples) if int(args.text_max_samples) > 0 else None),
                seed=int(args.seed),
            )
            text_dl = DataLoader(text_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=0)

    model = SeekerOmniLM(cfg.model).to(device)
    load_checkpoint(args.ckpt, model=model, optimizer=None)
    model.train()

    if not bool(args.unfreeze_llm):
        _freeze_llm_keep_mm_adapters(model)

    from transformers import AutoConfig, AutoImageProcessor, SiglipVisionModel

    vcfg = AutoConfig.from_pretrained(str(args.vision_model))
    if str(getattr(vcfg, "model_type", "")) != "siglip":
        raise SystemExit(
            f"vision_model must be a SigLIP/SigLIP2 checkpoint (model_type='siglip'), got {getattr(vcfg,'model_type',None)!r}"
        )
    processor = AutoImageProcessor.from_pretrained(str(args.vision_model))
    vision = SiglipVisionModel.from_pretrained(str(args.vision_model)).to(device)
    vision.train()
    if bool(args.freeze_vision):
        for p in vision.parameters():
            p.requires_grad = False
    elif int(args.vision_train_last_n) > 0:
        freeze_vision_all_but_last_n(vision, last_n=int(args.vision_train_last_n))

    patch_size = getattr(vision.config, "patch_size", None)
    if patch_size is None:
        raise SystemExit("could not infer vision patch_size from vision.config")
    patch_size = int(patch_size)

    feat_dim = int(getattr(vision.config, "hidden_size", 0) or 0)
    if feat_dim != int(cfg.model.image_feat_dim):
        raise SystemExit(f"vision feat_dim mismatch: vision={feat_dim} cfg.model.image_feat_dim={cfg.model.image_feat_dim}")

    resampler = PerceiverResampler(
        dim=int(feat_dim),
        num_latents=int(cfg.model.image_tokens),
        num_layers=int(args.resampler_layers),
        num_heads=int(args.resampler_heads),
        ff_mult=int(args.resampler_ff_mult),
    ).to(device)
    resampler.train()
    if str(args.image_embed) == "avgpool":
        for p in resampler.parameters():
            p.requires_grad = False

    teacher_vision = None
    if float(args.vision_distill_weight) > 0.0 and (not bool(args.freeze_vision)):
        teacher_vision = SiglipVisionModel.from_pretrained(str(args.vision_model)).to(device)
        teacher_vision.eval()
        for p in teacher_vision.parameters():
            p.requires_grad = False

    out_dir = Path(args.out_dir)
    vision_out_dir = Path(args.vision_out_dir)
    vision_out_dir.mkdir(parents=True, exist_ok=True)

    tb_writer = None
    tb_every = int(args.tb_every) if int(args.tb_every) > 0 else int(args.log_every)
    if bool(int(args.tb_enable)):
        from torch.utils.tensorboard import SummaryWriter

        tb_dir = Path(str(args.tb_dir)) if str(args.tb_dir).strip() else default_tb_dir(out_dir)
        tb_writer = SummaryWriter(log_dir=str(tb_dir))
        print(f"tb: {tb_dir}")
    (vision_out_dir / "vision_config.json").write_text(
        json.dumps(
            {
                "vision_model": str(args.vision_model),
                "patch_size": int(patch_size),
                "feat_dim": int(feat_dim),
                "freeze_vision": bool(args.freeze_vision),
                "vision_train_last_n": int(args.vision_train_last_n),
                "image_embed": str(args.image_embed),
                "vision_distill_weight": float(args.vision_distill_weight),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (vision_out_dir / "resampler_config.json").write_text(
        json.dumps(
            {
                "kind": "perceiver_resampler",
                "dim": int(feat_dim),
                "num_latents": int(cfg.model.image_tokens),
                "num_layers": int(args.resampler_layers),
                "num_heads": int(args.resampler_heads),
                "ff_mult": int(args.resampler_ff_mult),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lr_bridge = float(args.lr_bridge) if float(args.lr_bridge) > 0.0 else float(args.lr)
    lr_resampler = float(args.lr_resampler) if float(args.lr_resampler) > 0.0 else float(args.lr)
    lr_vision = float(args.lr_vision) if float(args.lr_vision) > 0.0 else float(args.lr)
    lr_llm = float(args.lr_llm) if float(args.lr_llm) > 0.0 else float(args.lr)

    bridge_params, llm_params = _split_model_param_groups(model)
    resampler_params = [p for p in resampler.parameters() if p.requires_grad]
    vision_params = [p for p in vision.parameters() if p.requires_grad]

    param_groups: list[dict] = []
    if bridge_params:
        param_groups.append({"params": bridge_params, "lr_base": lr_bridge, "name": "bridge"})
    if resampler_params:
        param_groups.append({"params": resampler_params, "lr_base": lr_resampler, "name": "resampler"})
    if vision_params:
        param_groups.append({"params": vision_params, "lr_base": lr_vision, "name": "vision"})
    if llm_params:
        param_groups.append({"params": llm_params, "lr_base": lr_llm, "name": "llm"})

    if not param_groups:
        raise SystemExit("no trainable parameters found (check freeze/unfreeze flags)")

    params: list[torch.nn.Parameter] = []
    for g in param_groups:
        params.extend(list(g["params"]))

    opt = torch.optim.AdamW(
        [{"params": g["params"], "lr": float(g["lr_base"]), "lr_base": float(g["lr_base"]), "name": str(g["name"])} for g in param_groups],
        weight_decay=float(args.weight_decay),
    )

    # 断点续训支持（仅 seeker ckpt；vision ckpt 若存在会一并加载）。
    start_step = 0
    ckpt = latest_checkpoint(out_dir)
    if ckpt is not None:
        start_step = load_checkpoint(ckpt, model=model, optimizer=(opt if bool(int(args.resume_optimizer)) else None))
        vpt = vision_out_dir / f"vision_step_{start_step:09d}.pt"
        if vpt.exists():
            vision.load_state_dict(torch.load(vpt, map_location="cpu"), strict=True)
        rpt = vision_out_dir / f"resampler_step_{start_step:09d}.pt"
        if rpt.exists():
            sd = torch.load(rpt, map_location="cpu")
            if not isinstance(sd, dict):
                raise TypeError(f"unexpected resampler checkpoint type: {type(sd)} ({rpt})")
            # avgpool 模式会存一个空 dict 作为标记；不要用 strict=True 加载。
            if len(sd) > 0:
                resampler.load_state_dict(sd, strict=True)
    else:
        # 新跑：打破“门控关闭 => img_proj 梯度为 0”的循环。
        with torch.no_grad():
            model.img_gate.fill_(1.0)

    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda" and dtype == torch.float16))

    step = int(start_step)
    total_steps = int(args.max_steps)
    t0 = time.time()

    pbar = tqdm(total=int(total_steps), initial=int(step), desc="train(e2e-vision-s2)")
    it = iter(dl)
    text_it = iter(text_dl) if text_dl is not None else None
    while step < int(total_steps):
        step += 1
        for g in opt.param_groups:
            base = float(g.get("lr_base", float(args.lr)))
            g["lr"] = cosine_lr(step, base_lr=base, total_steps=int(total_steps), warmup_steps=int(args.warmup_steps))

        use_text = False
        if text_it is not None and float(args.text_mix_ratio) > 0.0:
            r = float(args.text_mix_ratio)
            if not (0.0 <= r <= 1.0):
                raise SystemExit("--text_mix_ratio must be in [0,1]")
            use_text = random.random() < r

        if use_text:
            try:
                batch = next(text_it)
            except StopIteration:
                text_it = iter(text_dl)  # type: ignore[arg-type]
                batch = next(text_it)
        else:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl)
                batch = next(it)

        input_ids = batch["input_ids"].to(device=device, dtype=torch.long)
        labels = batch["labels"].to(device=device, dtype=torch.long)
        attention_mask = batch["attention_mask"].to(device=device, dtype=torch.float32)
        image_paths = list(batch["image_path"])

        has_image = bool(image_paths and str(image_paths[0]).strip())
        if has_image:
            imgs = [load_rgb(p) for p in image_paths]
            px = processor(images=imgs, return_tensors="pt").get("pixel_values")
            if px is None:
                raise RuntimeError("vision processor did not return pixel_values")
            px = px.to(device=device)
        else:
            px = None

        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type != "cpu" and dtype != torch.float32)):
            distill_loss = None
            if has_image:
                vout = vision(pixel_values=px)
                hs = vout.last_hidden_state  # [B, (1+P?) , D]

                h = int(px.shape[-2])  # type: ignore[union-attr]
                w = int(px.shape[-1])  # type: ignore[union-attr]
                patch_h = max(1, h // int(patch_size))
                patch_w = max(1, w // int(patch_size))
                patch_count = int(patch_h * patch_w)
                if int(hs.shape[1]) == int(patch_count) + 1:
                    hs = hs[:, 1:, :]

                if teacher_vision is not None:
                    with torch.no_grad():
                        tout = teacher_vision(pixel_values=px)
                        ths = tout.last_hidden_state
                        if int(ths.shape[1]) == int(patch_count) + 1:
                            ths = ths[:, 1:, :]
                    distill_loss = mse_distill(hs, ths)

                if str(args.image_embed) == "avgpool":
                    image_feats = pool_tokens_torch(hs, target_tokens=int(cfg.model.image_tokens))
                else:
                    image_feats = resampler(hs)

                if bool(int(args.mm_corrupt)):
                    input_ids = _corrupt_answer_tokens_for_mm(
                        input_ids,
                        labels,
                        unk_id=int(model.special.unk),
                        n_special=int(model.n_special),
                    )
            else:
                image_feats = None

            out = model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels,
                image_feats=image_feats,
            )

            loss = out.loss
            if distill_loss is not None and float(args.vision_distill_weight) > 0.0:
                loss = loss + float(args.vision_distill_weight) * distill_loss
            loss = loss / float(args.grad_accum)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % int(args.grad_accum) == 0:
            if scaler.is_enabled():
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, float(args.grad_clip))
            if scaler.is_enabled():
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)

        if step % int(args.log_every) == 0:
            dt = time.time() - t0
            ppl = float(torch.exp(out.loss.detach().float()).cpu())
            vdl = 0.0
            if has_image and distill_loss is not None:
                vdl = float(distill_loss.detach().float().cpu())
            pbar.set_postfix(loss=float(out.loss.detach().float().cpu()), ppl=ppl, vdl=vdl, sec=int(dt))

        if tb_writer is not None and int(tb_every) > 0 and (step % int(tb_every) == 0):
            with torch.no_grad():
                tb_writer.add_scalar("train/loss", float(out.loss.detach().float().cpu()), global_step=int(step))
                tb_writer.add_scalar("train/ppl", float(torch.exp(out.loss.detach().float()).cpu()), global_step=int(step))
                if has_image and distill_loss is not None:
                    tb_writer.add_scalar("train/vision_distill", float(distill_loss.detach().float().cpu()), global_step=int(step))
                tb_writer.add_scalar("train/lr", float(opt.param_groups[0]["lr"]), global_step=int(step))

        if step % int(args.save_every) == 0 or step == int(total_steps):
            save_checkpoint(
                out_dir,
                model=model,
                optimizer=opt,
                step=int(step),
                cfg=asdict(cfg),
                keep_last=int(args.keep_last),
            )
            vpt = vision_out_dir / f"vision_step_{step:09d}.pt"
            tmp = vpt.with_suffix(vpt.suffix + ".tmp")
            try:
                torch.save(vision.state_dict(), tmp)
                tmp.replace(vpt)
            finally:
                tmp.unlink(missing_ok=True)

            rpt = vision_out_dir / f"resampler_step_{step:09d}.pt"
            tmp = rpt.with_suffix(rpt.suffix + ".tmp")
            try:
                if str(args.image_embed) == "avgpool":
                    torch.save({}, tmp)
                else:
                    torch.save(resampler.state_dict(), tmp)
                tmp.replace(rpt)
            finally:
                tmp.unlink(missing_ok=True)

            if int(args.keep_last) > 0:
                pts = sorted(vision_out_dir.glob("vision_step_*.pt"))
                if len(pts) > int(args.keep_last):
                    for p in pts[: -int(args.keep_last)]:
                        p.unlink(missing_ok=True)

                pts = sorted(vision_out_dir.glob("resampler_step_*.pt"))
                if len(pts) > int(args.keep_last):
                    for p in pts[: -int(args.keep_last)]:
                        p.unlink(missing_ok=True)

        pbar.update(1)

    pbar.close()
    if tb_writer is not None:
        tb_writer.close()
    print(f"done. out_dir={out_dir} vision_out_dir={vision_out_dir}")
    return 0


def run_from_yaml_config(e2e_cfg: dict) -> int:
    exp_cfg = ExperimentConfig.load(e2e_cfg["config"])
    ckpt = _resolve_start_ckpt(exp_cfg, e2e_cfg)
    args = _args_from_yaml(e2e_cfg, ckpt)
    return run(args)
