import os
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk

from eval.gsm8k_reward import extract_gold_value
from eval.mmlu_pro_reward import extract_gold_choice


GSM8K_PROMPT_TEMPLATE = (
    "Solve the following math problem.\n"
    "Reason step by step.\n"
    "You must put the final numeric answer in exactly one boxed form like "
    "\\boxed{{123}}.\n"
    "Do not use any other final answer format.\n\n"
    "Question: {question}"
)

MATH_PROMPT_TEMPLATE = (
    "Solve the following math problem.\n"
    "Reason step by step.\n"
    "You must put the final answer in exactly one boxed form like "
    "\\boxed{{123}}.\n"
    "Do not use any other final answer format.\n\n"
    "Problem: {question}"
)

MMLU_PRO_PROMPT_TEMPLATE = (
    "Answer the following multiple-choice question.\n"
    "Reason step by step.\n"
    "You must put the final answer in exactly one boxed form like "
    "\\boxed{{A}}.\n"
    "Do not use any other final answer format.\n"
    "Only box a single option letter from A to J.\n\n"
    "Question: {question}\n\n"
    "Options:\n{options}"
)

_MATH_PROMPT_PREFIX = "Solve the following math problem step by step."
_MATH_PROMPT_SUFFIX = "Remember to put your answer on its own line after \"Answer:\"."
_MATH_SOURCES = {"math", "math_data"}
_MMLU_PRO_SOURCES = {"mmlu_pro", "mmlu-pro"}
_OPTION_LABELS = "ABCDEFGHIJ"


@contextmanager
def temporarily_unset_proxy_env() -> Iterator[None]:
    keys = ["all_proxy", "ALL_PROXY"]
    saved = {key: os.environ.pop(key) for key in keys if key in os.environ}
    try:
        yield
    finally:
        os.environ.update(saved)


def _normalize_source(source: str) -> str:
    return str(source or "gsm8k").strip().replace("-", "_").lower()


def _is_math_source(source: str) -> bool:
    return _normalize_source(source) in _MATH_SOURCES


def _is_mmlu_pro_source(source: str) -> bool:
    return _normalize_source(source) in _MMLU_PRO_SOURCES


def _format_multiple_choice_options(options: Sequence[str]) -> str:
    if not options:
        raise ValueError("multiple-choice prompts require at least one option")
    if len(options) > len(_OPTION_LABELS):
        raise ValueError(f"expected at most {len(_OPTION_LABELS)} options, got {len(options)}")
    rendered: list[str] = []
    for index, option in enumerate(options):
        rendered.append(f"{_OPTION_LABELS[index]}. {str(option).strip()}")
    return "\n".join(rendered)


def build_prompt(question: str, *, source: str = "gsm8k", options: Sequence[str] | None = None) -> str:
    normalized_source = _normalize_source(source)
    if _is_math_source(normalized_source):
        template = MATH_PROMPT_TEMPLATE
        return template.format(question=question.strip())
    if _is_mmlu_pro_source(normalized_source):
        return MMLU_PRO_PROMPT_TEMPLATE.format(
            question=question.strip(),
            options=_format_multiple_choice_options(options or []),
        )
    return GSM8K_PROMPT_TEMPLATE.format(question=question.strip())


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


def _process_split(split: Dataset, split_name: str) -> Dataset:
    def _map_fn(example: dict, idx: int) -> dict:
        question = example["question"].strip()
        answer = example["answer"]
        return {
            "id": f"{split_name}_{idx}",
            "split": split_name,
            "question": question,
            "gold_raw": answer,
            "gold_value": extract_gold_value(answer),
            "prompt": build_prompt(question, source="gsm8k"),
        }

    return split.map(_map_fn, with_indices=True, remove_columns=list(split.features))


def _extract_prompt_content(prompt_payload) -> str:
    if isinstance(prompt_payload, str):
        return prompt_payload.strip()
    if prompt_payload is None:
        return ""
    chunks: list[str] = []
    for item in prompt_payload:
        if isinstance(item, dict):
            content = item.get("content")
            if content:
                chunks.append(str(content).strip())
        elif item:
            chunks.append(str(item).strip())
    return "\n\n".join(chunk for chunk in chunks if chunk)


def _extract_math_question(prompt_payload, extra_info) -> str:
    if isinstance(extra_info, dict):
        raw_problem = extra_info.get("raw_problem")
        if isinstance(raw_problem, str) and raw_problem.strip():
            return raw_problem.strip()

    content = _extract_prompt_content(prompt_payload)
    if not content:
        return ""
    parts = [part.strip() for part in content.split("\n\n") if part.strip()]
    if parts and parts[0].startswith(_MATH_PROMPT_PREFIX):
        parts = parts[1:]
    if parts and parts[-1].startswith(_MATH_PROMPT_SUFFIX):
        parts = parts[:-1]
    question = "\n\n".join(parts).strip()
    return question or content


def _process_math_split(split: Dataset, split_name: str) -> Dataset:
    def _map_fn(example: dict, idx: int) -> dict:
        question = _extract_math_question(example.get("prompt"), example.get("extra_info"))
        reward_model = example.get("reward_model") or {}
        gold_raw = reward_model.get("ground_truth", "")
        data_source = example.get("data_source", "math_data")
        record_id = None
        extra_info = example.get("extra_info")
        if isinstance(extra_info, dict):
            record_id = extra_info.get("index")
        if record_id is None:
            record_id = idx
        return {
            "id": f"{split_name}_{record_id}",
            "split": split_name,
            "question": question,
            "gold_raw": gold_raw,
            "gold_value": extract_gold_value(gold_raw),
            "prompt": build_prompt(question, source="math_data"),
            "data_source": data_source,
        }

    return split.map(_map_fn, with_indices=True, remove_columns=list(split.features))


def _answer_index_to_letter(answer_index: int | None) -> str | None:
    if answer_index is None:
        return None
    if answer_index < 0 or answer_index >= len(_OPTION_LABELS):
        raise ValueError(f"answer_index out of range: {answer_index}")
    return _OPTION_LABELS[answer_index]


def _process_mmlu_pro_split(split: Dataset, split_name: str) -> Dataset:
    def _map_fn(example: dict, idx: int) -> dict:
        question = str(example["question"]).strip()
        options = [str(option).strip() for option in example.get("options", [])]
        raw_answer = str(example.get("answer") or "").strip()
        answer_index_raw = example.get("answer_index")
        answer_index = int(answer_index_raw) if answer_index_raw is not None else None
        gold_value = extract_gold_choice(raw_answer or _answer_index_to_letter(answer_index) or "")
        question_id = example.get("question_id")
        record_id = question_id if question_id is not None else idx
        return {
            "id": f"{split_name}_{record_id}",
            "split": split_name,
            "question": question,
            "options": options,
            "category": str(example.get("category") or "").strip(),
            "src": str(example.get("src") or "").strip(),
            "gold_raw": raw_answer or gold_value,
            "gold_value": gold_value,
            "answer_index": answer_index if answer_index is not None else _OPTION_LABELS.index(gold_value),
            "prompt": build_prompt(question, source="mmlu_pro", options=options),
            "data_source": "mmlu_pro",
        }

    return split.map(_map_fn, with_indices=True, remove_columns=list(split.features))


def _allocate_split_counts(group_sizes: dict[str, int], target_total: int) -> dict[str, int]:
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


def _split_dataset_three_way(
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
    test_allocations = _allocate_split_counts(group_sizes, test_target)
    remaining_group_sizes = {key: group_sizes[key] - test_allocations[key] for key in group_sizes}
    val_allocations = _allocate_split_counts(remaining_group_sizes, val_target)

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


def ensure_raw_dataset(raw_path: str | Path, *, source: str = "gsm8k") -> DatasetDict:
    raw_path = Path(raw_path)
    if raw_path.exists():
        dataset = load_from_disk(str(raw_path))
        if not isinstance(dataset, DatasetDict):
            raise TypeError(f"expected DatasetDict at {raw_path}, got {type(dataset)}")
        return dataset

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with temporarily_unset_proxy_env():
        normalized_source = _normalize_source(source)
        if _is_mmlu_pro_source(normalized_source):
            dataset = load_dataset("TIGER-Lab/MMLU-Pro")
        else:
            dataset = load_dataset("openai/gsm8k", "main")
    dataset.save_to_disk(str(raw_path))
    return dataset


def ensure_processed_dataset(
    *,
    raw_path: str | Path,
    processed_path: str | Path,
    split_seed: int,
    val_size: int,
    source: str = "gsm8k",
    mmlu_pro_raw_splits: Sequence[str] | None = None,
    mmlu_pro_train_ratio: float = 0.8,
    mmlu_pro_val_ratio: float = 0.1,
    mmlu_pro_test_ratio: float = 0.1,
) -> DatasetDict:
    processed_path = Path(processed_path)
    if processed_path.exists():
        dataset = load_from_disk(str(processed_path))
        if not isinstance(dataset, DatasetDict):
            raise TypeError(f"expected DatasetDict at {processed_path}, got {type(dataset)}")
        return dataset

    normalized_source = _normalize_source(source)
    if _is_math_source(normalized_source):
        raw_dir = Path(raw_path)
        processed = DatasetDict(
            {
                "train": _process_math_split(Dataset.from_parquet(str(raw_dir / "dapo-math-17k.parquet")), "train"),
                "val": _process_math_split(Dataset.from_parquet(str(raw_dir / "validation.parquet")), "val"),
                "test": _process_math_split(Dataset.from_parquet(str(raw_dir / "test.parquet")), "test"),
            }
        )
    elif _is_mmlu_pro_source(normalized_source):
        raw_dataset = ensure_raw_dataset(raw_path, source=normalized_source)
        raw_split_names = tuple(mmlu_pro_raw_splits or ("validation", "test"))
        missing_splits = [name for name in raw_split_names if name not in raw_dataset]
        if missing_splits:
            raise KeyError(f"MMLU-Pro raw splits not found: {missing_splits}")
        combined = concatenate_datasets([raw_dataset[name] for name in raw_split_names]).shuffle(seed=split_seed)
        train_split, val_split, test_split = _split_dataset_three_way(
            combined,
            split_seed=split_seed,
            train_ratio=float(mmlu_pro_train_ratio),
            val_ratio=float(mmlu_pro_val_ratio),
            test_ratio=float(mmlu_pro_test_ratio),
            stratify_column="category",
        )
        processed = DatasetDict(
            {
                "train": _process_mmlu_pro_split(train_split, "train"),
                "val": _process_mmlu_pro_split(val_split, "val"),
                "test": _process_mmlu_pro_split(test_split, "test"),
            }
        )
    else:
        raw_dataset = ensure_raw_dataset(raw_path, source=normalized_source)
        train_split = raw_dataset["train"].train_test_split(test_size=val_size, seed=split_seed)
        processed = DatasetDict(
            {
                "train": _process_split(train_split["train"], "train"),
                "val": _process_split(train_split["test"], "val"),
                "test": _process_split(raw_dataset["test"], "test"),
            }
        )
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed.save_to_disk(str(processed_path))
    return processed


def load_records(dataset: Dataset, max_examples: int = 0) -> list[dict]:
    if max_examples and max_examples > 0:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
    return [dataset[idx] for idx in range(len(dataset))]
