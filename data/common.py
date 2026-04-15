from __future__ import annotations

import os
import random
from contextlib import contextmanager
from typing import Any, Iterator

from datasets import Dataset


_DEFAULT_SOURCE = "gsm8k"
_BENCHMARK_ALIASES = {
    "gsm8k": "gsm8k",
    "mmlu_pro": "mmlu-pro",
    "math500": "math500",
    "olympiadbench": "olympiadbench",
    "minerva": "minerva",
    "amc23": "amc23",
    "aime2024": "aime2024",
    "aime2025": "aime2025",
    "idavidrein/gpqa": "gpqa",
    "gpqa": "gpqa",
    "digitallearninggmbh/math_lighteval": "math-lighteval",
    "math_lighteval": "math-lighteval",
    "dapo_math_17k": "dapo-math-17k",
}


@contextmanager
def temporarily_unset_proxy_env() -> Iterator[None]:
    keys = ["all_proxy", "ALL_PROXY"]
    saved = {key: os.environ.pop(key) for key in keys if key in os.environ}
    try:
        yield
    finally:
        os.environ.update(saved)


def normalize_source(source: str | None) -> str:
    return str(source or _DEFAULT_SOURCE).strip().replace("-", "_").lower()


def canonical_benchmark_name(source: str | None, default: str = _DEFAULT_SOURCE) -> str:
    raw_source = str(source or default).strip()
    normalized = normalize_source(raw_source)
    if normalized in _BENCHMARK_ALIASES:
        return _BENCHMARK_ALIASES[normalized]
    if "/" in raw_source:
        suffix = normalize_source(raw_source.rsplit("/", 1)[-1])
        if suffix in _BENCHMARK_ALIASES:
            return _BENCHMARK_ALIASES[suffix]
        return suffix
    return normalized


def apply_chat_template_to_prompt(
    prompt: str,
    *,
    tokenizer: Any,
    system_message: str | None = None,
    enable_thinking: bool | None = None,
) -> str:
    messages: list[dict[str, str]] = []
    if system_message and system_message.strip():
        messages.append({"role": "system", "content": system_message.strip()})
    messages.append({"role": "user", "content": prompt.strip()})

    kwargs: dict[str, Any] = {
        "conversation": messages,
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(**kwargs)


def apply_chat_template_to_records(
    records: list[dict],
    *,
    tokenizer: Any,
    system_message: str | None = None,
    enable_thinking: bool | None = None,
) -> list[dict]:
    formatted: list[dict] = []
    for record in records:
        updated = dict(record)
        updated["prompt"] = apply_chat_template_to_prompt(
            record["prompt"],
            tokenizer=tokenizer,
            system_message=system_message,
            enable_thinking=enable_thinking,
        )
        formatted.append(updated)
    return formatted


def load_records(dataset: Dataset, max_examples: int = 0) -> list[dict]:
    if max_examples and max_examples > 0:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
    return [dataset[idx] for idx in range(len(dataset))]


def export_split_datasets(
    split_datasets: dict[str, Dataset],
    *,
    export_dir: str | os.PathLike[str],
) -> None:
    export_root = os.fspath(export_dir)
    os.makedirs(export_root, exist_ok=True)
    for split_name, dataset in split_datasets.items():
        dataset.to_parquet(os.path.join(export_root, f"{split_name}.parquet"))


def allocate_split_counts(group_sizes: dict[str, int], target_total: int) -> dict[str, int]:
    total = sum(group_sizes.values())
    if target_total <= 0 or total <= 0:
        return {key: 0 for key in group_sizes}

    allocations = {key: 0 for key in group_sizes}
    remainders: list[tuple[float, str]] = []
    assigned = 0
    for key, group_size in group_sizes.items():
        desired = target_total * group_size / total
        base = min(group_size, int(desired))
        allocations[key] = base
        assigned += base
        remainders.append((desired - base, key))

    remaining = min(target_total - assigned, total - assigned)
    for _, key in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        if allocations[key] >= group_sizes[key]:
            continue
        allocations[key] += 1
        remaining -= 1
    return allocations


def split_dataset_three_way(
    dataset: Dataset,
    *,
    split_seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    stratify_column: str | None = None,
) -> tuple[Dataset, Dataset, Dataset]:
    if train_ratio < 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("split ratios must be non-negative")
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("at least one split ratio must be positive")

    train_ratio /= total_ratio
    val_ratio /= total_ratio
    test_ratio /= total_ratio

    total_examples = len(dataset)
    train_target = int(round(total_examples * train_ratio))
    val_target = int(round(total_examples * val_ratio))
    test_target = total_examples - train_target - val_target
    if test_target < 0:
        test_target = 0
        val_target = max(0, total_examples - train_target)
    if train_target <= 0 or val_target < 0 or test_target < 0:
        raise ValueError(
            f"invalid split sizes resolved from ratios: train={train_target}, val={val_target}, test={test_target}"
        )

    if stratify_column is None or stratify_column not in dataset.column_names:
        empty = dataset.select([])
        if train_ratio == 1.0:
            return dataset, empty, empty
        if train_ratio == 0.0:
            train_split = empty
            remainder = dataset
        else:
            initial = dataset.train_test_split(test_size=1.0 - train_ratio, seed=split_seed)
            train_split = initial["train"]
            remainder = initial["test"]

        holdout_total = val_ratio + test_ratio
        if holdout_total == 0.0:
            return train_split, empty, empty
        if val_ratio == 0.0:
            return train_split, empty, remainder
        if test_ratio == 0.0:
            return train_split, remainder, empty

        relative_test_ratio = test_ratio / holdout_total
        final = remainder.train_test_split(test_size=relative_test_ratio, seed=split_seed)
        return train_split, final["train"], final["test"]

    grouped_indices: dict[str, list[int]] = {}
    for index, value in enumerate(dataset[stratify_column]):
        key = str(value or "__missing__")
        grouped_indices.setdefault(key, []).append(index)

    rng = random.Random(split_seed)
    for indices in grouped_indices.values():
        rng.shuffle(indices)

    group_sizes = {key: len(indices) for key, indices in grouped_indices.items()}
    test_allocations = allocate_split_counts(group_sizes, test_target)
    remaining_group_sizes = {key: group_sizes[key] - test_allocations[key] for key in group_sizes}
    val_allocations = allocate_split_counts(remaining_group_sizes, val_target)

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    for key in sorted(grouped_indices):
        indices = grouped_indices[key]
        test_n = test_allocations[key]
        val_n = val_allocations[key]
        test_indices.extend(indices[:test_n])
        val_indices.extend(indices[test_n : test_n + val_n])
        train_indices.extend(indices[test_n + val_n :])

    train_indices.sort()
    val_indices.sort()
    test_indices.sort()
    return dataset.select(train_indices), dataset.select(val_indices), dataset.select(test_indices)
