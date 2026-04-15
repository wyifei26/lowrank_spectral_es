from __future__ import annotations

import argparse
import os

from config_utils import apply_overrides, load_raw_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lowrank / spectral ES on math QA datasets.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--mode", choices=["train", "baseline"], default="train")
    parser.add_argument("--baseline-split", default="test")
    parser.add_argument("--baseline-max-examples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_raw_yaml_config(args.config), args.override)
    trainer = None
    try:
        algorithm_name = config.get("algorithm", {}).get("name")
        if algorithm_name != "spectral_es":
            raise ValueError(f"only spectral_es is supported, got {algorithm_name}")
        backend = config.get("execution", {}).get("backend")
        distributed_mode = config.get("execution", {}).get("distributed_mode")
        if backend != "vllm" or distributed_mode != "mutant_parallel":
            raise ValueError(
                "only vLLM single-node multi-GPU mutant-parallel execution is supported; "
                f"got backend={backend}, distributed_mode={distributed_mode}"
            )
        # We launch one vLLM engine per torchrun rank. Tell vLLM to use its
        # torchrun-compatible external_launcher path so each rank binds to its
        # own LOCAL_RANK device instead of defaulting every in-process engine
        # to cuda:0.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        from engine.distributed_vllm_trainer import DistributedVLLMSpectralESTrainer

        trainer = DistributedVLLMSpectralESTrainer(config)
        if args.mode == "baseline":
            trainer.run_baseline(split=args.baseline_split, max_examples=args.baseline_max_examples)
            return
        trainer.run()
    finally:
        if trainer is not None:
            trainer.close()


if __name__ == "__main__":
    main()
