from __future__ import annotations

from pathlib import Path
from typing import Sequence

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk

from data.common import export_split_datasets, split_dataset_three_way, temporarily_unset_proxy_env
from eval.mmlu_pro_reward import extract_gold_choice


PROMPT_TEMPLATE = (
    "Answer the following multiple-choice question.\n"
    "Reason step by step.\n"
    "You must put the final answer in exactly one boxed form like "
    "\\boxed{{A}}.\n"
    "Only box a single option letter from A to J.\n\n"
    "Question: {question}\n\n"
    "Options:\n{options}"
)
OPTION_LABELS = "ABCDEFGHIJ"


def _format_multiple_choice_options(options: Sequence[str]) -> str:
    if not options:
        raise ValueError("multiple-choice prompts require at least one option")
    if len(options) > len(OPTION_LABELS):
        raise ValueError(f"expected at most {len(OPTION_LABELS)} options, got {len(options)}")
    rendered: list[str] = []
    for index, option in enumerate(options):
        rendered.append(f"{OPTION_LABELS[index]}. {str(option).strip()}")
    return "\n".join(rendered)


def build_prompt(question: str, *, options: Sequence[str]) -> str:
    return PROMPT_TEMPLATE.format(
        question=question.strip(),
        options=_format_multiple_choice_options(options),
    )


def _answer_index_to_letter(answer_index: int | None) -> str | None:
    if answer_index is None:
        return None
    if answer_index < 0 or answer_index >= len(OPTION_LABELS):
        raise ValueError(f"answer_index out of range: {answer_index}")
    return OPTION_LABELS[answer_index]


def process_split(split: Dataset, split_name: str) -> Dataset:
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
            "answer_index": answer_index if answer_index is not None else OPTION_LABELS.index(gold_value),
            "prompt": build_prompt(question, options=options),
            "data_source": "mmlu_pro",
        }

    return split.map(_map_fn, with_indices=True, remove_columns=list(split.features))


def ensure_raw_dataset(raw_path: str | Path) -> DatasetDict:
    raw_path = Path(raw_path)
    if (raw_path / "dataset_dict.json").exists():
        dataset = load_from_disk(str(raw_path))
        if not isinstance(dataset, DatasetDict):
            raise TypeError(f"expected DatasetDict at {raw_path}, got {type(dataset)}")
        return dataset

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with temporarily_unset_proxy_env():
        dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    dataset.save_to_disk(str(raw_path))
    return dataset


def ensure_processed_dataset(
    *,
    raw_path: str | Path,
    processed_path: str | Path,
    split_seed: int,
    val_size: int,
    raw_splits: Sequence[str] | None = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> DatasetDict:
    del val_size
    processed_path = Path(processed_path)
    if (processed_path / "dataset_dict.json").exists():
        dataset = load_from_disk(str(processed_path))
        if not isinstance(dataset, DatasetDict):
            raise TypeError(f"expected DatasetDict at {processed_path}, got {type(dataset)}")
        return dataset

    raw_dataset = ensure_raw_dataset(raw_path)
    raw_split_names = tuple(raw_splits or ("validation", "test"))
    missing_splits = [name for name in raw_split_names if name not in raw_dataset]
    if missing_splits:
        raise KeyError(f"MMLU-Pro raw splits not found: {missing_splits}")
    combined = concatenate_datasets([raw_dataset[name] for name in raw_split_names]).shuffle(seed=split_seed)
    train_split, val_split, test_split = split_dataset_three_way(
        combined,
        split_seed=split_seed,
        train_ratio=float(train_ratio),
        val_ratio=float(val_ratio),
        test_ratio=float(test_ratio),
        stratify_column="category",
    )
    processed = DatasetDict(
        {
            "train": process_split(train_split, "train"),
            "val": process_split(val_split, "val"),
            "test": process_split(test_split, "test"),
        }
    )
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed.save_to_disk(str(processed_path))
    export_split_datasets(dict(processed), export_dir=processed_path.parent / "processed_exports")
    return processed
