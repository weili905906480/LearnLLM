from ..config import TrainConfig
from ..model.lm import SeekerOmniLM


def apply_stage_freeze(model: SeekerOmniLM, train_cfg: TrainConfig) -> None:
    for p in model.parameters():
        p.requires_grad = True

    freeze_backbone = bool(train_cfg.freeze_backbone)
    unfreeze_last_n = int(train_cfg.unfreeze_last_n_layers or 0)

    if freeze_backbone:
        for p in model.blocks.parameters():
            p.requires_grad = False
        for p in model.norm.parameters():
            p.requires_grad = False
        for p in model.base_embed.parameters():
            p.requires_grad = False

        if unfreeze_last_n > 0:
            n = min(int(unfreeze_last_n), len(model.blocks))
            for blk in model.blocks[-n:]:
                for p in blk.parameters():
                    p.requires_grad = True
            for p in model.norm.parameters():
                p.requires_grad = True

    if bool(train_cfg.freeze_base_embed):
        for p in model.base_embed.parameters():
            p.requires_grad = False

    if bool(train_cfg.freeze_special_embed):
        for p in model.special_embed.parameters():
            p.requires_grad = False
