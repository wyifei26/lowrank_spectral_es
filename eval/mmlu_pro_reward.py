from __future__ import annotations

from eval.answer_parser import extract_choice_letter_answer, normalize_choice_letter
from eval.gsm8k_reward import DEFAULT_REWARD_CONFIG, RewardResult


def extract_gold_choice(answer: str, *, answer_index: int | None = None) -> str:
    normalized = normalize_choice_letter(str(answer)) if answer is not None else None
    if normalized is not None:
        return normalized
    if answer_index is not None:
        if answer_index < 0 or answer_index >= 10:
            raise ValueError(f"answer_index out of range for MMLU-Pro: {answer_index}")
        return chr(ord("A") + answer_index)
    raise ValueError(f"could not normalize MMLU-Pro gold answer: {answer}")


def score_prediction(
    prediction: str,
    gold_value: str,
    *,
    reward_config: dict | None = None,
) -> RewardResult:
    reward_config = {**DEFAULT_REWARD_CONFIG, **(reward_config or {})}
    predicted_value = extract_choice_letter_answer(prediction)
    correct = predicted_value is not None and predicted_value == gold_value
    reward = reward_config["exact_match"] * float(correct)
    return RewardResult(
        reward=reward,
        predicted_value=predicted_value,
        gold_value=gold_value,
        correct=correct,
    )
