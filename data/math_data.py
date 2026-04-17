from __future__ import annotations

from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk

from data.common import export_split_datasets
from eval.gsm8k_reward import extract_gold_value


PROMPT_TEMPLATE = (
    "Solve the following question.\n"
    "Reason step by step, and provide the final answer in \\boxed{{}} format.\n"
    "Question: {question}"
)
_PROMPT_PREFIX = "Solve the following math problem step by step."
_PROMPT_SUFFIX = 'Remember to put your answer on its own line after "Answer:".'


def build_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question.strip())


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
    if parts and parts[0].startswith(_PROMPT_PREFIX):
        parts = parts[1:]
    if parts and parts[-1].startswith(_PROMPT_SUFFIX):
        parts = parts[:-1]
    question = "\n\n".join(parts).strip()
    return question or content


def process_split(split: Dataset, split_name: str) -> Dataset:
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
            "prompt": build_prompt(question),
            "data_source": data_source,
        }

    return split.map(_map_fn, with_indices=True, remove_columns=list(split.features))


def ensure_processed_dataset(
    *,
    raw_path: str | Path,
    processed_path: str | Path,
    split_seed: int,
    val_size: int,
) -> DatasetDict:
    del split_seed, val_size
    processed_path = Path(processed_path)
    if (processed_path / "dataset_dict.json").exists():
        dataset = load_from_disk(str(processed_path))
        if not isinstance(dataset, DatasetDict):
            raise TypeError(f"expected DatasetDict at {processed_path}, got {type(dataset)}")
        return dataset

    raw_dir = Path(raw_path)
    processed = DatasetDict(
        {
            "train": process_split(Dataset.from_parquet(str(raw_dir / "dapo-math-17k.parquet")), "train"),
            "val": process_split(Dataset.from_parquet(str(raw_dir / "validation.parquet")), "val"),
            "test": process_split(Dataset.from_parquet(str(raw_dir / "test.parquet")), "test"),
        }
    )
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed.save_to_disk(str(processed_path))
    export_split_datasets(dict(processed), export_dir=processed_path.parent / "processed_exports")
    return processed
