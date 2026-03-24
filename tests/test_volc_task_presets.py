from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from volc_task_presets import (  # noqa: E402
    DATASET_PRESETS,
    MODEL_PRESETS,
    PARAM_PRESETS,
    SAMPLING_PRESETS,
    TaskSelection,
    build_overrides,
    resolve_task_selection,
)


def test_resolve_task_selection_derives_4gpu_runtime_values() -> None:
    selection = TaskSelection(
        dataset=DATASET_PRESETS["gsm8k"],
        model=MODEL_PRESETS["qwen3_0p6b_base"],
        params=PARAM_PRESETS["default"],
        sampling=SAMPLING_PRESETS["pairwise_default"],
        gpu_count=4,
        wandb_group="spectral_es",
    )

    resolved = resolve_task_selection(selection)

    assert resolved.world_size == 4
    assert resolved.gpus_per_node == 4
    assert resolved.flavor == "ml.pni2.14xlarge"
    assert resolved.mutants_per_worker == 16
    assert resolved.mutant_chunk_size == 16
    assert resolved.train_micro_batch == 64


def test_resolve_task_selection_derives_8gpu_runtime_values() -> None:
    selection = TaskSelection(
        dataset=DATASET_PRESETS["mmlu_pro"],
        model=MODEL_PRESETS["qwen3_8b"],
        params=PARAM_PRESETS["default"],
        sampling=SAMPLING_PRESETS["pairwise_default"],
        gpu_count=8,
        wandb_group="spectral_es_mmlu_pro",
    )

    resolved = resolve_task_selection(selection)

    assert resolved.world_size == 8
    assert resolved.gpus_per_node == 8
    assert resolved.flavor == "ml.pni2.28xlarge"
    assert resolved.mutants_per_worker == 4
    assert resolved.mutant_chunk_size == 4
    assert resolved.train_micro_batch == 64


def test_build_overrides_uses_derived_runtime_values() -> None:
    selection = TaskSelection(
        dataset=DATASET_PRESETS["gsm8k"],
        model=MODEL_PRESETS["qwen3_0p6b_base"],
        params=PARAM_PRESETS["default"],
        sampling=SAMPLING_PRESETS["gaussian_mean_default"],
        gpu_count=4,
        wandb_group="spectral_es",
    )

    overrides = build_overrides(resolve_task_selection(selection), run_id="demo_run")

    assert "execution.mutants_per_worker=16" in overrides
    assert "execution.mutant_chunk_size=16" in overrides
    assert "train.micro_batch=64" in overrides
    assert "train.effective_question_batch=64" in overrides
    assert "es.update_rule=gaussian_mean" in overrides
    assert "es.antithetic=false" in overrides
    assert "es.sigma.m=0.001" in overrides


def test_build_overrides_supports_per_layer_cma_sampling() -> None:
    selection = TaskSelection(
        dataset=DATASET_PRESETS["mmlu_pro"],
        model=MODEL_PRESETS["qwen3_8b"],
        params=PARAM_PRESETS["default"],
        sampling=SAMPLING_PRESETS["per_layer_cma"],
        gpu_count=8,
        wandb_group="spectral_es_cma",
    )

    overrides = build_overrides(resolve_task_selection(selection), run_id="demo_cma")

    assert "es.update_rule=per_layer_cma_es" in overrides
    assert "es.antithetic=true" in overrides
    assert "es.sigma.m=0.001" in overrides


def test_resolve_task_selection_rejects_invalid_gpu_count() -> None:
    selection = TaskSelection(
        dataset=DATASET_PRESETS["gsm8k"],
        model=MODEL_PRESETS["qwen3_0p6b_base"],
        params=PARAM_PRESETS["default"],
        sampling=SAMPLING_PRESETS["pairwise_default"],
        gpu_count=2,
        wandb_group="spectral_es",
    )

    with pytest.raises(ValueError, match="gpu_count must be one of"):
        resolve_task_selection(selection)
