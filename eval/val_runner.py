from __future__ import annotations

import json
from pathlib import Path

from engine.batch_executor import BatchExecutor


def write_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def evaluate_current_state(
    *,
    executor: BatchExecutor,
    records: list[dict],
    question_micro_batch: int,
    output_dir: str | Path | None = None,
    collect_predictions: bool = True,
) -> dict:
    result = executor.score_active_mutants(
        records=records,
        num_mutants=1,
        question_micro_batch=question_micro_batch,
        collect_predictions=collect_predictions,
    )
    accuracy = float(result.exact_match_rates[0].item()) if result.exact_match_rates.numel() else 0.0
    summary = {
        "num_examples": len(records),
        "accuracy": accuracy,
        "reward_mean": float(result.rewards[0].item()) if result.rewards.numel() else 0.0,
        "profiler": result.profiler_snapshot,
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        if collect_predictions:
            write_jsonl(output_dir / "predictions.jsonl", result.predictions)
            summary["predictions_path"] = str(output_dir / "predictions.jsonl")
        write_json(output_dir / "summary.json", summary)
    return summary
