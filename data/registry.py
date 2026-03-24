from __future__ import annotations

from typing import Sequence

from datasets import DatasetDict

import data.gsm8k as gsm8k
import data.math_data as math_data
import data.mmlu_pro as mmlu_pro
from data.common import normalize_source


_MATH_SOURCES = {"math", "math_data"}
_MMLU_PRO_SOURCES = {"mmlu_pro"}


def build_prompt(question: str, *, source: str = "gsm8k", options: Sequence[str] | None = None) -> str:
    normalized_source = normalize_source(source)
    if normalized_source in _MATH_SOURCES:
        return math_data.build_prompt(question)
    if normalized_source in _MMLU_PRO_SOURCES:
        return mmlu_pro.build_prompt(question, options=options or [])
    return gsm8k.build_prompt(question)


def ensure_processed_dataset(
    *,
    raw_path: str,
    processed_path: str,
    split_seed: int,
    val_size: int,
    source: str = "gsm8k",
    mmlu_pro_raw_splits: Sequence[str] | None = None,
    mmlu_pro_train_ratio: float = 0.8,
    mmlu_pro_val_ratio: float = 0.1,
    mmlu_pro_test_ratio: float = 0.1,
) -> DatasetDict:
    normalized_source = normalize_source(source)
    if normalized_source in _MATH_SOURCES:
        return math_data.ensure_processed_dataset(
            raw_path=raw_path,
            processed_path=processed_path,
            split_seed=split_seed,
            val_size=val_size,
        )
    if normalized_source in _MMLU_PRO_SOURCES:
        return mmlu_pro.ensure_processed_dataset(
            raw_path=raw_path,
            processed_path=processed_path,
            split_seed=split_seed,
            val_size=val_size,
            raw_splits=mmlu_pro_raw_splits,
            train_ratio=mmlu_pro_train_ratio,
            val_ratio=mmlu_pro_val_ratio,
            test_ratio=mmlu_pro_test_ratio,
        )
    return gsm8k.ensure_processed_dataset(
        raw_path=raw_path,
        processed_path=processed_path,
        split_seed=split_seed,
        val_size=val_size,
    )
