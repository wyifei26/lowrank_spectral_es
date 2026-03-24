#!/usr/bin/env python3

from __future__ import annotations

from volc_task_presets import (
    DATASET_PRESETS,
    MODEL_PRESETS,
    PARAM_PRESETS,
    TaskSelection,
    preview_submission,
    submit_config,
    write_yaml,
)


DEFAULT_SELECTION = TaskSelection(
    dataset=DATASET_PRESETS["mmlu_pro"],
    model=MODEL_PRESETS["qwen3_8b"],
    params=PARAM_PRESETS["default"],
    gpu_count=4,
    wandb_group="spectral_es_mmlu_pro",
)


def main() -> None:
    _, _, _, config_path, payload = preview_submission(DEFAULT_SELECTION)
    write_yaml(config_path, payload)

    print(f"Wrote config: {config_path}")
    task_id = submit_config(config_path)
    print(f"Submitted task: {task_id}")
    print(f"Watch status: volc ml_task get --id {task_id}")
    print(f"Watch logs:   volc ml_task logs --task {task_id} --instance worker_0 --lines 200")


if __name__ == "__main__":
    main()
