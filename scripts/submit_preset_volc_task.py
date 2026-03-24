#!/usr/bin/env python3

from __future__ import annotations

from volc_task_presets import (
    DATASET_PRESETS,
    MODEL_PRESETS,
    PARAM_PRESETS,
    SAMPLING_PRESETS,
    TaskSelection,
    preview_submission,
    submit_config,
    write_yaml,
)


# Edit these five values before running the script.
DATASET_KEY = "gsm8k"
MODEL_KEY = "qwen3_0p6b_nothink"
PARAM_KEY = "default"
SAMPLING_KEY = "cma"
GPU_COUNT = 4

# Optional knobs for the generated task.
WANDB_GROUP = "spectral_es_manual"


def main() -> None:
    selection = TaskSelection(
        dataset=DATASET_PRESETS[DATASET_KEY],
        model=MODEL_PRESETS[MODEL_KEY],
        params=PARAM_PRESETS[PARAM_KEY],
        sampling=SAMPLING_PRESETS[SAMPLING_KEY],
        gpu_count=GPU_COUNT,
        wandb_group=WANDB_GROUP,
    )
    _, _, _, config_path, payload = preview_submission(selection)
    write_yaml(config_path, payload)

    print(f"Wrote config: {config_path}")
    user_input = input("Press Enter to submit this task, or type anything to cancel: ")
    if user_input.strip():
        print("Submission cancelled.")
        return

    task_id = submit_config(config_path)
    print(f"Submitted task: {task_id}")
    print(f"Watch status: volc ml_task get --id {task_id}")
    print(f"Watch logs:   volc ml_task logs --task {task_id} --instance worker_0 --lines 200")


if __name__ == "__main__":
    main()
