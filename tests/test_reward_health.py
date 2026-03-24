from eval.gsm8k_reward import extract_gold_value, score_prediction


def test_score_prediction_is_binary_exact_match_reward():
    correct = score_prediction("The answer is \\boxed{5}", "5")
    wrong = score_prediction("The answer is \\boxed{4}", "5")
    malformed = score_prediction("I think it is five", "5")

    assert correct.correct is True
    assert correct.reward == 1.0

    assert wrong.correct is False
    assert wrong.reward == 0.0

    assert malformed.correct is False
    assert malformed.reward == 0.0


def test_extract_gold_value_supports_plain_math_ground_truth():
    assert extract_gold_value("33") == "33"
    assert extract_gold_value("#### 44") == "44"
    assert extract_gold_value("B") == "b"
