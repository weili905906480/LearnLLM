from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

_MINIMIND_CHAT_TEMPLATE = """{%- if tools %}
    {{- '<|im_start|>system\\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\\n\\n' }}
    {%- endif %}
    {{- "# Tools\\n\\nYou may call one or more functions to assist the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}
    {%- for tool in tools %}
        {{- "\\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}
{%- else %}
 {%- if messages[0]['role'] == 'system' -%}
        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}
    {%- else -%}
        {{- '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}
 {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}
    {%- elif message.role == "assistant" %}
   {{- '<|im_start|>' + message.role + '\\n' + content }}
  {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\\n{\\"name\\": \\"' }}
                {{- tool_call.name }}
                {{- '\\", \\"arguments\\": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\\n<tool_response>\\n' }}
        {{- content }}
        {{- '\\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- endif %}
{%- endif %}"""


def _infer_core_tokens(special_tokens: list[str]) -> tuple[str, str, str, str, str]:
    """从 special_tokens 推断 scheme 名称与 pad/bos/eos/unk 的字符串形式。

    当前 dataprep 只支持 minimind2_chatml。
    """

    s = set(map(str, special_tokens))

    required = {"<|endoftext|>", "<|im_start|>", "<|im_end|>"}
    if not required.issubset(s):
        raise ValueError(
            "special_tokens must include minimind2_chatml core tokens: "
            "<|endoftext|>, <|im_start|>, <|im_end|>"
        )

    scheme = "minimind2_chatml"
    pad = "<|endoftext|>"
    bos = "<|im_start|>"
    eos = "<|im_end|>"
    unk = "<|endoftext|>"
    return scheme, pad, bos, eos, unk


def _write_tokenizer_config(
    *,
    out_dir: Path,
    tok: Tokenizer,
    special_tokens: list[str],
    scheme_name: str,
    pad_token: str,
    bos_token: str,
    eos_token: str,
    unk_token: str,
) -> None:
    added_tokens_decoder = {}
    for i, t in enumerate(special_tokens):
        added_tokens_decoder[str(i)] = {
            "content": t,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        }

    cfg = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "clean_up_tokenization_spaces": False,
        "legacy": True,
        "model_max_length": 32768,
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "added_tokens_decoder": added_tokens_decoder,
        "additional_special_tokens": [],
        "bos_token": str(bos_token),
        "eos_token": str(eos_token),
        "pad_token": str(pad_token),
        "unk_token": str(unk_token),
        "vocab_size": int(tok.get_vocab_size()),
    }
    if str(scheme_name) == "minimind2_chatml":
        cfg["chat_template"] = _MINIMIND_CHAT_TEMPLATE

    (out_dir / "tokenizer_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def train_text_bpe(
    *,
    input_path: str | Path,
    out_dir: str | Path,
    vocab_size: int = 6400,
    special_tokens: list[str],
) -> None:
    inp = Path(input_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    special_tokens = list(special_tokens)
    scheme_name, pad_token, bos_token, eos_token, unk_token = _infer_core_tokens(special_tokens)

    tok = Tokenizer(BPE(unk_token=str(unk_token), byte_fallback=True))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)

    trainer = BpeTrainer(
        vocab_size=int(vocab_size),
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet(),
    )

    tok.train([str(inp)], trainer=trainer)

    # 保证特殊 token 的 ID 稳定（SeekerOmniLM 的词嵌入拆分依赖这一点）。
    for i, t in enumerate(special_tokens):
        tid = tok.token_to_id(t)
        if tid is None or int(tid) != int(i):
            raise ValueError(f"special token id mismatch: {t} -> {tid} (expected {i})")
    tok.decoder = ByteLevelDecoder()

    tok_path = out / "tokenizer.json"
    tok.save(str(tok_path))
    _write_tokenizer_config(
        out_dir=out,
        tok=tok,
        special_tokens=special_tokens,
        scheme_name=str(scheme_name),
        pad_token=str(pad_token),
        bos_token=str(bos_token),
        eos_token=str(eos_token),
        unk_token=str(unk_token),
    )

    meta = {
        "special_tokens_scheme": str(scheme_name),
        "vocab_size": tok.get_vocab_size(),
        "special_tokens": special_tokens,
        "special_ids": {t: tok.token_to_id(t) for t in special_tokens},
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_text_bpe(tokenizer_dir: str | Path) -> Tokenizer:
    p = Path(tokenizer_dir) / "tokenizer.json"
    return Tokenizer.from_file(str(p))
