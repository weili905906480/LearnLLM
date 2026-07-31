from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenSchemeSpec:
    name: str
    special_tokens: list[str]

    pad_token: str
    bos_token: str
    eos_token: str
    unk_token: str


MINIMIND2_CHATML = TokenSchemeSpec(
    name="minimind2_chatml",
    special_tokens=[
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<img_bos>",
        "<img>",
        "<img_eos>",
    ],
    pad_token="<|endoftext|>",
    bos_token="<|im_start|>",
    eos_token="<|im_end|>",
    unk_token="<|endoftext|>",
)


_SPECS: dict[str, TokenSchemeSpec] = {
    MINIMIND2_CHATML.name: MINIMIND2_CHATML,
}


def get_token_scheme_spec(scheme: str | None) -> TokenSchemeSpec:
    scheme = (scheme or MINIMIND2_CHATML.name).strip()
    spec = _SPECS.get(scheme)
    if spec is None:
        opts = ", ".join(sorted(_SPECS.keys()))
        raise ValueError(f"unknown special_tokens_scheme={scheme!r} (expected one of: {opts})")
    return spec


@dataclass(frozen=True)
class SpecialTokenIds:
    pad: int
    unk: int
    bos: int
    eos: int

    img_bos: int
    img: int
    img_eos: int


def build_special_token_ids(spec: TokenSchemeSpec) -> SpecialTokenIds:
    tok2id = {t: i for i, t in enumerate(spec.special_tokens)}

    def _id(t: str) -> int:
        if t not in tok2id:
            raise ValueError(f"required special token missing from scheme={spec.name!r}: {t}")
        return int(tok2id[t])

    return SpecialTokenIds(
        pad=_id(spec.pad_token),
        unk=_id(spec.unk_token),
        bos=_id(spec.bos_token),
        eos=_id(spec.eos_token),
        img_bos=_id("<img_bos>"),
        img=_id("<img>"),
        img_eos=_id("<img_eos>"),
    )


DEFAULT_SPECIAL_TOKENS_SCHEME = MINIMIND2_CHATML.name
