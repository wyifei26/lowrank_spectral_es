from __future__ import annotations

from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from data.common import temporarily_unset_proxy_env
from eval.gsm8k_reward import extract_gold_value


PROMPT_TEMPLATE = (
    "Solve the following math problem.\n"
    "Reason step by step.\n"
    "You must put the final numeric answer in exactly one boxed form like "
    "\\boxed{{123}}.\n\n"
    "Question: {question}"
)


def build_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question.strip())


def process_split(split: Dataset, split_name: str) -> Dataset:
    def _map_fn(example: dict, idx: int) -> dict:
        question = example["question"].strip()
        answer = example["answer"]
        return {
            "id": f"{split_name}_{idx}",
            "split": split_name,
            "question": question,
            "gold_raw": answer,
            "gold_value": extract_gold_value(answer),
            "prompt": build_prompt(question),
        }

    return split.map(_map_fn, with_indices=True, remove_columns=list(split.features))


def ensure_raw_dataset(raw_path: str | Path) -> DatasetDict:
    raw_path = Path(raw_path)
    if raw_path.exists():
        dataset = load_from_disk(str(raw_path))
        if not isinstance(dataset, DatasetDict):
            raise TypeError(f"expected DatasetDict at {raw_path}, got {type(dataset)}")
        return dataset

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with temporarily_unset_proxy_env():
        dataset = load_dataset("openai/gsm8k", "main")
    dataset.save_to_disk(str(raw_path))
    return dataset


def ensure_processed_dataset(
    *,
    raw_path: str | Path,
    processed_path: str | Path,
    split_seed: int,
    val_size: int,
) -> DatasetDict:
    processed_path = Path(processed_path)
    if processed_path.exists():
        dataset = load_from_disk(str(processed_path))
        if not isinstance(dataset, DatasetDict):
            raise TypeError(f"expected DatasetDict at {processed_path}, got {type(dataset)}")
        return dataset

    raw_dataset = ensure_raw_dataset(raw_path)
    train_split = raw_dataset["train"].train_test_split(test_size=val_size, seed=split_seed)
    processed = DatasetDict(
        {
            "train": process_split(train_split["train"], "train"),
            "val": process_split(train_split["test"], "val"),
            "test": process_split(raw_dataset["test"], "test"),
        }
    )
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed.save_to_disk(str(processed_path))
    return processed
