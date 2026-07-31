import torch


def inject_feature_tokens(
    x: torch.Tensor,
    *,
    input_ids: torch.Tensor,
    image_feats: torch.Tensor | None,
    img_token_id: int,
    img_proj: torch.nn.Module,
    img_gate: torch.Tensor,
) -> torch.Tensor:
    # x: [B,S,H]
    if image_feats is not None:
        img_mask = input_ids == int(img_token_id)
        if img_mask.any():
            img_tokens = img_proj(image_feats.to(dtype=x.dtype))
            img_tokens = img_tokens * torch.tanh(img_gate)[None, None, :]
            pos = img_mask.nonzero(as_tuple=False)  # [N,2] row-major
            flat = img_tokens.reshape(-1, img_tokens.shape[-1])
            if pos.shape[0] == flat.shape[0]:
                x[pos[:, 0], pos[:, 1]] = x[pos[:, 0], pos[:, 1]] + flat
            else:
                for b in range(int(x.shape[0])):
                    idx = torch.where(img_mask[b])[0]
                    if idx.numel() == 0:
                        continue
                    take = min(idx.numel(), img_tokens.shape[1])
                    x[b, idx[:take]] = x[b, idx[:take]] + img_tokens[b, :take]

    return x
