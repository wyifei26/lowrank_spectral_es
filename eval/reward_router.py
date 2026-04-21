from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

from eval.answer_parser import extract_normalized_boxed_answer, normalize_answer_string
from eval.gsm8k_reward import DEFAULT_REWARD_CONFIG, RewardResult, score_prediction as score_gsm8k_prediction
from eval.mmlu_pro_reward import score_prediction as score_mmlu_pro_prediction


_MATH_SOURCES = {"math", "math_data"}
_MMLU_PRO_SOURCES = {"mmlu_pro", "mmlu-pro"}


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_verifier_path() -> Path:
    env_path = os.environ.get("MATH_VERIFIER_PATH")
    if env_path:
        return Path(env_path)

    candidates = [
        _workspace_root() / "code" / "verl" / "verifier.py",
        _workspace_root() / "verl" / "verifier.py",
        Path(__file__).resolve().parents[1] / "verl" / "verifier.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _normalize_source_name(source: str | None) -> str:
    return str(source or "gsm8k").strip().replace("-", "_").lower()


@lru_cache(maxsize=None)
def _load_verl_verifier_from_path(path_str: str) -> ModuleType:
    verifier_path = Path(path_str)
    if not verifier_path.exists():
        raise FileNotFoundError(f"math verifier not found at {verifier_path}")
    spec = importlib.util.spec_from_file_location("lowrank_spectral_es_rl_verl_verifier", verifier_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load verifier spec from {verifier_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_verl_verifier(verifier_path: str | None = None) -> ModuleType:
    return _load_verl_verifier_from_path(str(Path(verifier_path) if verifier_path else _default_verifier_path()))


def _resolve_source(record: dict[str, Any], reward_config: dict[str, Any] | None = None) -> str:
    reward_config = reward_config or {}
    source = reward_config.get("data_source") or record.get("data_source") or "gsm8k"
    return _normalize_source_name(source)


def _score_math_prediction(
    prediction: str,
    record: dict[str, Any],
    *,
    reward_config: dict[str, Any] | None = None,
) -> RewardResult:
    reward_config = {**DEFAULT_REWARD_CONFIG, **(reward_config or {})}
    verifier = _load_verl_verifier(reward_config.get("math_verifier_path"))
    ground_truth = str(record.get("gold_raw") or record.get("gold_value") or "").strip()
    predicted_value = None
    extract_answer = getattr(verifier, "extract_answer", None)
    if callable(extract_answer):
        predicted_value = extract_answer(prediction)
    if predicted_value is None:
        predicted_value = extract_normalized_boxed_answer(prediction)
    normalize_math_answer = getattr(verifier, "mathd_normalize_answer", None)
    if callable(normalize_math_answer) and predicted_value is not None:
        predicted_value = normalize_math_answer(predicted_value)

    score = verifier.compute_score_no_think(
        _resolve_source(record, reward_config),
        prediction,
        ground_truth,
        extra_info=record.get("extra_info"),
    )
    correct = float(score) > 0.0
    reward = reward_config["exact_match"] * float(correct)
    gold_value = ground_truth
    if callable(normalize_math_answer):
        gold_value = normalize_math_answer(ground_truth)
    else:
        gold_value = normalize_answer_string(ground_truth) or ground_truth
    return RewardResult(
        reward=reward,
        predicted_value=predicted_value,
        gold_value=gold_value,
        correct=correct,
    )


def score_record_prediction(
    prediction: str,
    record: dict[str, Any],
    *,
    reward_config: dict[str, Any] | None = None,
) -> RewardResult:
    source = _resolve_source(record, reward_config)
    if source in _MATH_SOURCES:
        return _score_math_prediction(prediction, record, reward_config=reward_config)
    if source in _MMLU_PRO_SOURCES:
        return score_mmlu_pro_prediction(
            prediction,
            record["gold_value"],
            reward_config=reward_config,
        )
    return score_gsm8k_prediction(
        prediction,
        record["gold_value"],
        reward_config=reward_config,
    )
