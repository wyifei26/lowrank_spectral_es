from types import SimpleNamespace

from eval.reward_router import score_record_prediction


def test_score_record_prediction_keeps_gsm8k_on_lightweight_verifier():
    record = {
        "id": "gsm8k_0",
        "question": "Compute 2+3.",
        "gold_value": "5",
    }

    result = score_record_prediction("The answer is \\boxed{5}.", record)

    assert result.correct is True
    assert result.reward == 1.0
    assert result.predicted_value == "5"
    assert result.gold_value == "5"


def test_score_record_prediction_uses_verl_verifier_for_math_data(monkeypatch):
    calls: list[tuple[str, str, str]] = []

    def _compute_score_no_think(data_source, solution_str, ground_truth, extra_info=None, **_kwargs):
        calls.append((data_source, solution_str, ground_truth))
        return 1.0 if ground_truth == r"\frac{1}{2}" and r"\boxed{\frac{1}{2}}" in solution_str else -1.0

    fake_verifier = SimpleNamespace(
        compute_score_no_think=_compute_score_no_think,
        extract_answer=lambda text: r"\frac{1}{2}" if r"\boxed{\frac{1}{2}}" in text else None,
        mathd_normalize_answer=lambda text: text.replace(" ", "") if text is not None else None,
    )
    monkeypatch.setattr("eval.reward_router._load_verl_verifier", lambda verifier_path=None: fake_verifier)

    record = {
        "id": "math_0",
        "question": "What is one half?",
        "gold_value": "2",
        "gold_raw": r"\frac{1}{2}",
        "data_source": "math_data",
    }

    result = score_record_prediction("We conclude \\boxed{\\frac{1}{2}}.", record)

    assert calls == [("math_data", "We conclude \\boxed{\\frac{1}{2}}.", r"\frac{1}{2}")]
    assert result.correct is True
    assert result.reward == 1.0
    assert result.predicted_value == r"\frac{1}{2}"
    assert result.gold_value == r"\frac{1}{2}"


def test_score_record_prediction_supports_mmlu_pro():
    record = {
        "id": "mmlu_pro_0",
        "question": "Which option is correct?",
        "gold_value": "B",
        "data_source": "mmlu_pro",
    }

    result = score_record_prediction("Let's reason it through.\n\nFinal answer: \\boxed{B}", record)

    assert result.correct is True
    assert result.reward == 1.0
    assert result.predicted_value == "B"
    assert result.gold_value == "B"
