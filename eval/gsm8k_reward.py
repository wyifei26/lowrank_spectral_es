from __future__ import annotations

from dataclasses import dataclass

from eval.answer_parser import extract_normalized_boxed_answer, normalize_answer_string


DEFAULT_REWARD_CONFIG = {
    "exact_match": 1.0,
}


@dataclass
class RewardResult:
    reward: float
    predicted_value: str | None
    gold_value: str
    correct: bool


def extract_gold_value(answer: str) -> str:
    marker = "####"
    candidate = answer.split(marker)[-1].strip() if marker in answer else str(answer).strip()
    normalized = normalize_answer_string(candidate)
    if normalized is None:
        raise ValueError(f"could not normalize gold answer: {answer}")
    return normalized


def score_prediction(
    prediction: str,
    gold_value: str,
    *,
    reward_config: dict | None = None,
) -> RewardResult:
    reward_config = {**DEFAULT_REWARD_CONFIG, **(reward_config or {})}
    predicted_value = extract_normalized_boxed_answer(prediction)
    correct = predicted_value is not None and predicted_value == gold_value
    reward = reward_config["exact_match"] * float(correct)
    return RewardResult(
        reward=reward,
        predicted_value=predicted_value,
        gold_value=gold_value,
        correct=correct,
    )
