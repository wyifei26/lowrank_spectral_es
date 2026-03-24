from __future__ import annotations


def summarize_single_mutant_result(result) -> dict[str, float]:
    if result.rewards.numel() == 0:
        return {
            "reward_mean": 0.0,
            "accuracy": 0.0,
        }
    return {
        "reward_mean": float(result.rewards[0].item()),
        "accuracy": float(result.exact_match_rates[0].item()),
    }
