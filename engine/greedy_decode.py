from __future__ import annotations

import torch

from models.mutant_linear import MutantModel


@torch.no_grad()
def greedy_generate_texts(
    *,
    model: MutantModel,
    tokenizer,
    prompts: list[str],
    mutant_indices: torch.Tensor,
    max_new_tokens: int,
) -> tuple[list[str], int]:
    device = model.device
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    mutant_indices = mutant_indices.to(device)
    eos_token_id = tokenizer.eos_token_id

    generated_tokens: list[torch.Tensor] = []
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=device)
    past_key_values = None
    current_input_ids = input_ids
    current_attention_mask = attention_mask

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=current_input_ids,
            attention_mask=current_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            mutant_indices=mutant_indices,
        )
        logits = outputs.logits[:, -1, :]
        next_tokens = torch.argmax(logits, dim=-1)
        next_tokens = torch.where(finished, torch.full_like(next_tokens, eos_token_id), next_tokens)
        generated_tokens.append(next_tokens)
        finished = finished | next_tokens.eq(eos_token_id)
        if finished.all():
            break
        past_key_values = outputs.past_key_values
        current_input_ids = next_tokens.unsqueeze(1)
        current_attention_mask = torch.cat(
            [
                current_attention_mask,
                torch.ones((current_attention_mask.shape[0], 1), dtype=current_attention_mask.dtype, device=device),
            ],
            dim=1,
        )

    if generated_tokens:
        generated_ids = torch.stack(generated_tokens, dim=1)
        texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        token_count = int(generated_ids.numel())
    else:
        texts = [""] * len(prompts)
        token_count = 0
    return texts, token_count
