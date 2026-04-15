from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_utils import load_yaml_config
from engine.distributed_vllm_trainer import DistributedVLLMSpectralESTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a plain HF model directory on GSM8K.")
    parser.add_argument("--config", required=True, help="Resolved training config YAML path.")
    parser.add_argument("--model-path", required=True, help="HF model directory to evaluate.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--run-id", required=True, help="Output run id for this evaluation.")
    parser.add_argument("--baseline-output-dir", default="/GenSIvePFS/users/yfwang/output/gsm8k_eval")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", dest="top_p", type=float, default=None)
    parser.add_argument("--top-k", dest="top_k", type=int, default=None)
    parser.add_argument("--presence-penalty", dest="presence_penalty", type=float, default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def ensure_single_process_env() -> None:
    defaults = {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_WORLD_SIZE": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29533",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main() -> None:
    args = parse_args()
    ensure_single_process_env()

    config = load_yaml_config(Path(args.config))
    config["model"]["model_path"] = str(Path(args.model_path).resolve())

    config.setdefault("execution", {})
    config["execution"]["world_size"] = 1
    config["execution"]["gpus_per_node"] = 1
    config["execution"]["mutants_per_worker"] = int(config["es"]["num_mutants"])

    config["output"]["run_id"] = args.run_id
    config.setdefault("baseline", {})
    config["baseline"]["output_dir"] = args.baseline_output_dir
    config.setdefault("vllm", {})
    config["vllm"]["gpu_memory_utilization"] = float(args.gpu_memory_utilization)

    config.setdefault("generation", {})
    if args.temperature is not None:
        config["generation"]["temperature"] = float(args.temperature)
    if args.top_p is not None:
        config["generation"]["top_p"] = float(args.top_p)
    if args.top_k is not None:
        config["generation"]["top_k"] = int(args.top_k)
    if args.presence_penalty is not None:
        config["generation"]["presence_penalty"] = float(args.presence_penalty)

    if args.disable_wandb:
        config.setdefault("wandb", {})
        config["wandb"]["enabled"] = False
        config["wandb"]["name"] = args.run_id

    # Keep evaluation behavior aligned with the original base-model setup.
    config.setdefault("prompt", {})
    config["prompt"]["use_chat_template"] = False

    # Baseline eval does not use adapters; shrink SVD setup to reduce startup cost.
    config.setdefault("layers", {})
    config["layers"]["target_blocks"] = [0]
    config["layers"]["target_modules"] = ["q_proj"]

    trainer = None
    try:
        trainer = DistributedVLLMSpectralESTrainer(config)
        summary = trainer.run_baseline(split=args.split, max_examples=int(args.max_examples))
        output_dir = Path(config["output"]["root_dir"]) / args.run_id / "eval_outputs" / args.split / "step_0000"
        print(f"EVAL_OUTPUT_DIR {output_dir}", flush=True)
        print(f"FINAL_SUMMARY {summary}", flush=True)
    finally:
        if trainer is not None:
            trainer.close()


if __name__ == "__main__":
    main()
