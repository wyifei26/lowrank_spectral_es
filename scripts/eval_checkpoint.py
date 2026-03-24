from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from config_utils import load_yaml_config
from engine.distributed_vllm_trainer import DistributedVLLMSpectralESTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved spectral ES checkpoint on GSM8K.")
    parser.add_argument("--config", required=True, help="Resolved training config YAML path.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pt path containing adapter_state.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--run-id", required=True, help="Output run id for this evaluation.")
    parser.add_argument("--baseline-output-dir", default="/GenSIvePFS/users/yfwang/output/gsm8k_eval")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def ensure_single_process_env() -> None:
    defaults = {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_WORLD_SIZE": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29531",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main() -> None:
    args = parse_args()
    ensure_single_process_env()

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    config = load_yaml_config(config_path)

    config.setdefault("execution", {})
    config["execution"]["world_size"] = 1
    config["execution"]["gpus_per_node"] = 1
    config["execution"]["mutants_per_worker"] = int(config["es"]["num_mutants"])
    config["output"]["run_id"] = args.run_id
    config.setdefault("baseline", {})
    config["baseline"]["output_dir"] = args.baseline_output_dir
    config.setdefault("vllm", {})
    config["vllm"]["gpu_memory_utilization"] = float(args.gpu_memory_utilization)
    if args.disable_wandb:
        config.setdefault("wandb", {})
        config["wandb"]["enabled"] = False
        config["wandb"]["name"] = args.run_id

    trainer = None
    try:
        trainer = DistributedVLLMSpectralESTrainer(config)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        trainer.state.load_adapter_state_dict(checkpoint["adapter_state"])
        print(
            f"CHECKPOINT_LOADED step={checkpoint['step']} best_val_accuracy={checkpoint['best_val_accuracy']}",
            flush=True,
        )
        output_dir = Path(args.baseline_output_dir) / args.run_id / args.split
        summary = trainer._evaluate_split(
            split=args.split,
            step=int(checkpoint["step"]),
            max_examples=int(args.max_examples),
            output_dir=output_dir,
        )
        print(f"EVAL_OUTPUT_DIR {output_dir}", flush=True)
        print(f"FINAL_SUMMARY {summary}", flush=True)
    finally:
        if trainer is not None:
            trainer.close()


if __name__ == "__main__":
    main()
