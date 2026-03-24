from data.gsm8k import _extract_math_question, _split_dataset_three_way, apply_chat_template_to_prompt, build_prompt
from engine.distributed_vllm_trainer import _is_base_model
from eval.answer_parser import (
    extract_choice_letter_answer,
    extract_normalized_boxed_answer,
    normalize_answer_string,
    normalize_choice_letter,
    normalize_numeric_string,
)
from datasets import Dataset


def test_extract_last_boxed_value():
    text = "reasoning \\box{12} more reasoning \\boxed{34}"
    assert extract_normalized_boxed_answer(text) == "34"


def test_normalize_numeric_string():
    assert normalize_numeric_string("$1,200.00") == "1200"
    assert normalize_numeric_string("-0.50") == "-0.5"
    assert normalize_numeric_string("answer=72") == "72"


def test_normalize_answer_string_supports_textual_answers():
    assert normalize_answer_string(" B ") == "b"
    assert normalize_answer_string("${A}$") == "a"
    assert extract_normalized_boxed_answer("final \\box{C}") == "c"


def test_extract_math_question_removes_old_answer_format_instructions():
    prompt_payload = [
        {
            "role": "user",
            "content": (
                "Solve the following math problem step by step. The last line of your response should "
                "be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
                "Compute 2+3.\n\n"
                "Remember to put your answer on its own line after \"Answer:\"."
            ),
        }
    ]
    assert _extract_math_question(prompt_payload, extra_info={}) == "Compute 2+3."


def test_build_math_prompt_requires_box_answer():
    prompt = build_prompt("Compute 2+3.", source="math_data")
    assert "\\boxed{123}" in prompt
    assert "Do not use any other final answer format." in prompt


def test_build_mmlu_pro_prompt_renders_options():
    prompt = build_prompt(
        "Which option is correct?",
        source="mmlu_pro",
        options=["Alpha", "Beta", "Gamma"],
    )
    assert "\\boxed{A}" in prompt
    assert "A. Alpha" in prompt
    assert "C. Gamma" in prompt


def test_extract_choice_letter_answer_supports_boxed_and_plain_formats():
    assert normalize_choice_letter(" b ") == "B"
    assert extract_choice_letter_answer("Therefore the answer is \\boxed{C}.") == "C"
    assert extract_choice_letter_answer("After checking each option, the correct answer is (d).") == "D"


def test_split_dataset_three_way_can_stratify_by_category():
    dataset = Dataset.from_dict(
        {
            "question": [f"q{i}" for i in range(20)],
            "category": ["math"] * 10 + ["history"] * 10,
        }
    )
    train_split, val_split, test_split = _split_dataset_three_way(
        dataset,
        split_seed=7,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        stratify_column="category",
    )

    assert len(train_split) == 12
    assert len(val_split) == 4
    assert len(test_split) == 4
    assert train_split["category"].count("math") == 6
    assert val_split["category"].count("math") == 2
    assert test_split["category"].count("history") == 2


class _FakeTokenizer:
    def apply_chat_template(self, conversation, tokenize, add_generation_prompt, enable_thinking=None):
        assert tokenize is False
        assert add_generation_prompt is True
        rendered = []
        for message in conversation:
            rendered.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>")
        if enable_thinking is False:
            rendered.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        else:
            rendered.append("<|im_start|>assistant\n")
        return "\n".join(rendered)


def test_apply_chat_template_to_prompt_wraps_user_prompt():
    rendered = apply_chat_template_to_prompt(
        "Compute 2+3.",
        tokenizer=_FakeTokenizer(),
        system_message="You are a helpful math solver.",
    )
    assert "<|im_start|>system" in rendered
    assert "<|im_start|>user" in rendered
    assert "Compute 2+3." in rendered
    assert rendered.endswith("<|im_start|>assistant\n")


def test_is_base_model_detects_base_variants():
    assert _is_base_model("/tmp/Qwen3-0.6B-Base")
    assert _is_base_model("/tmp/some_model_base")
    assert not _is_base_model("/tmp/Qwen3-0.6B")
