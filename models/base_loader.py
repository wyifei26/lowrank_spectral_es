from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise KeyError(f"unsupported dtype name: {dtype_name}")
    return mapping[dtype_name]


def load_tokenizer(model_path: str | Path):
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(
    model_path: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
):
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model
