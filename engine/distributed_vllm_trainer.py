from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import torch
import torch.distributed as dist
import wandb
from transformers import PreTrainedTokenizerBase
from vllm import LLM

from config_utils import dump_yaml_config
from data import apply_chat_template_to_records, ensure_processed_dataset, load_records
from engine.distributed_utils import (
    MutantShard,
    aggregate_distributed_step_metrics,
    merge_sharded_rewards,
    resolve_mutant_shard,
)
from engine.gpu_monitor import GPUMonitor, GPUMonitorSnapshot
from engine.vllm_executor import VLLMSpectralExecutor
from es.cma import PerLayerCMAES
from es.noise import seed_everything
from es.spectral_update import (
    apply_alpha_update_to_direction_payloads,
    compute_gaussian_direction_payloads,
    compute_pairwise_direction_payloads,
)
from es.updater import resolve_named_value
from eval.health import summarize_single_mutant_result
from eval.val_runner import write_json, write_jsonl
from models.base_loader import load_causal_lm, load_tokenizer, resolve_dtype
from models.layer_selector import select_target_layers
from models.spectral_vllm import (
    PARAMETERIZATION_FULL_FACTORIZED_M,
    build_vllm_spectral_state,
    cleanup_cpu_model,
)
from models.svd_cache import load_or_create_svd_cache, resolve_svd_cache_path


BEIJING_TZ = ZoneInfo("Asia/Shanghai")
CMA_UPDATE_RULES = {"per_layer_diagonal_cma_es"}


def _patch_transformers_tokenizer_compat() -> None:
    if hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        return

    @property
    def _all_special_tokens_extended(self):
        return list(self.all_special_tokens)

    PreTrainedTokenizerBase.all_special_tokens_extended = _all_special_tokens_extended


def _is_base_model(model_path: str | os.PathLike[str]) -> bool:
    model_name = Path(model_path).name.lower()
    return model_name.endswith("-base") or model_name.endswith("_base")


class DistributedVLLMSpectralESTrainer:
    TRAIN_SAMPLE_PREVIEW_LIMIT = 16

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        required_envs = ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
        missing_envs = [name for name in required_envs if name not in os.environ]
        if missing_envs:
            raise RuntimeError(
                "mutant_parallel mode must be launched with torchrun; missing env vars: "
                + ", ".join(missing_envs)
            )
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        execution_config = config.setdefault("execution", {})
        configured_world_size_raw = execution_config.get("world_size")
        env_world_size = int(os.environ["WORLD_SIZE"])
        if configured_world_size_raw not in (None, 0):
            configured_world_size = int(configured_world_size_raw)
        else:
            configured_world_size = env_world_size
        if configured_world_size != env_world_size:
            raise ValueError(
                f"config.execution.world_size ({configured_world_size}) must match WORLD_SIZE ({env_world_size})"
            )
        self.world_size = env_world_size
        configured_gpus_per_node_raw = execution_config.get("gpus_per_node")
        env_gpus_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", str(self.world_size)))
        if configured_gpus_per_node_raw not in (None, 0):
            configured_gpus_per_node = int(configured_gpus_per_node_raw)
        else:
            configured_gpus_per_node = env_gpus_per_node
        if configured_gpus_per_node != env_gpus_per_node:
            raise ValueError(
                f"config.execution.gpus_per_node ({configured_gpus_per_node}) must match "
                f"LOCAL_WORLD_SIZE ({env_gpus_per_node})"
            )
        execution_config["world_size"] = self.world_size
        execution_config["gpus_per_node"] = env_gpus_per_node
        self.is_main_process = self.rank == 0
        self._init_process_group()

        self.run_id = config["output"]["run_id"]
        self.output_dir = Path(config["output"]["root_dir"]) / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.adapter_root = self.output_dir / "vllm_adapters"
        self.adapter_root.mkdir(parents=True, exist_ok=True)
        self.train_sample_root = self.output_dir / "train_samples"
        self.train_sample_root.mkdir(parents=True, exist_ok=True)
        self.eval_output_root = self.output_dir / "eval_outputs"
        self.eval_output_root.mkdir(parents=True, exist_ok=True)
        self.wandb_run = None
        self.gpu_monitor = GPUMonitor(device_index=self.local_rank)
        seed_everything(int(config["seed"]) + self.rank)
        self._init_wandb()

        mutants_per_worker_raw = execution_config.get("mutants_per_worker")
        self.current_k = int(config["es"]["num_mutants"])
        if mutants_per_worker_raw in (None, 0):
            if self.current_k % self.world_size != 0:
                raise ValueError(
                    f"cannot infer execution.mutants_per_worker because es.num_mutants ({self.current_k}) "
                    f"is not divisible by WORLD_SIZE ({self.world_size})"
                )
            self.mutants_per_worker = self.current_k // self.world_size
            execution_config["mutants_per_worker"] = self.mutants_per_worker
        else:
            self.mutants_per_worker = int(mutants_per_worker_raw)
        self._validate_execution_config()
        self.local_shard = resolve_mutant_shard(
            num_mutants=self.current_k,
            world_size=self.world_size,
            rank=self.rank,
            mutants_per_worker=self.mutants_per_worker,
        )

        self.tokenizer = load_tokenizer(config["model"]["model_path"])

        dataset = ensure_processed_dataset(
            raw_path=config["data"]["raw_dir"],
            processed_path=config["data"]["processed_dir"],
            split_seed=int(config["data"]["split_seed"]),
            val_size=int(config["data"].get("val_size", 0)),
            source=str(config["data"].get("source", "gsm8k")),
            mmlu_pro_raw_splits=config["data"].get("raw_splits"),
            mmlu_pro_train_ratio=float(config["data"].get("train_ratio", 0.8)),
            mmlu_pro_val_ratio=float(config["data"].get("val_ratio", 0.1)),
            mmlu_pro_test_ratio=float(config["data"].get("test_ratio", 0.1)),
        )
        self.train_records = load_records(dataset["train"], int(config["data"].get("train_max_examples", 0)))
        self.val_records = load_records(dataset["val"], int(config["data"].get("val_max_examples", 0)))
        self.test_records = load_records(dataset["test"], int(config["data"].get("test_max_examples", 0)))
        prompt_config = config.get("prompt", {})
        use_chat_template = bool(prompt_config.get("use_chat_template", False))
        if use_chat_template and not _is_base_model(config["model"]["model_path"]):
            system_message = prompt_config.get("system_message")
            enable_thinking = prompt_config.get("enable_thinking")
            self.train_records = apply_chat_template_to_records(
                self.train_records,
                tokenizer=self.tokenizer,
                system_message=system_message,
                enable_thinking=enable_thinking,
            )
            self.val_records = apply_chat_template_to_records(
                self.val_records,
                tokenizer=self.tokenizer,
                system_message=system_message,
                enable_thinking=enable_thinking,
            )
            self.test_records = apply_chat_template_to_records(
                self.test_records,
                tokenizer=self.tokenizer,
                system_message=system_message,
                enable_thinking=enable_thinking,
            )

        self._ensure_svd_cache_ready_on_gpu()

        cpu_model = load_causal_lm(
            config["model"]["model_path"],
            device=torch.device("cpu"),
            dtype=resolve_dtype(config["model"].get("svd_dtype", "float32")),
        )
        selections = select_target_layers(
            cpu_model,
            target_blocks=list(config["layers"]["target_blocks"]),
            target_modules=list(config["layers"]["target_modules"]),
        )
        cache_path, cache_payload = load_or_create_svd_cache(
            model_path=config["model"]["model_path"],
            selections=selections,
            rank=self._resolved_svd_rank(),
            band_strategy=config["subspace"]["band_strategy"],
            cache_dir=config["subspace"]["cache_dir"],
            device="cpu",
            compute_dtype=resolve_dtype(config["model"].get("svd_dtype", "float32")),
        )
        self.svd_cache_path = cache_path
        self.spectral_rank = max((int(bundle["u"].shape[1]) for bundle in cache_payload.values()), default=0)
        config["subspace"]["effective_rank"] = self.spectral_rank
        self.state = build_vllm_spectral_state(
            algorithm_name=config["algorithm"]["name"],
            selections=selections,
            cache_payload=cache_payload,
            subspace_config=config.get("subspace", {}),
        )
        config["subspace"]["effective_adapter_rank"] = int(self.state.export_rank)
        update_rule = config.get("es", {}).get("update_rule", "per_layer_diagonal_cma_es")
        self.cma_state: PerLayerCMAES | None = None
        if update_rule in CMA_UPDATE_RULES:
            self.cma_state = PerLayerCMAES(
                layer_shapes={name: adapter.m_state.shape for name, adapter in self.state.adapters.items()},
                sigma_config=config["es"]["sigma"],
                cma_config=dict(config.get("es", {}).get("cma", {})),
            )
        cleanup_cpu_model(cpu_model)

        vllm_config = config.get("vllm", {})
        _patch_transformers_tokenizer_compat()
        llm_kwargs: dict[str, Any] = {
            "model": config["model"]["model_path"],
            "tokenizer": config["model"]["model_path"],
            "trust_remote_code": True,
            "dtype": config["model"]["dtype"],
            "seed": int(config["seed"]),
            "enable_lora": True,
            "tensor_parallel_size": 1,
            "distributed_executor_backend": "external_launcher",
            "max_lora_rank": max(int(self.state.export_rank), 1),
            "max_loras": int(vllm_config.get("max_loras", config["execution"]["mutant_chunk_size"])),
            "max_cpu_loras": int(vllm_config.get("max_cpu_loras", config["execution"]["mutant_chunk_size"] * 2)),
            "gpu_memory_utilization": float(vllm_config.get("gpu_memory_utilization", 0.85)),
            "max_model_len": int(vllm_config.get("max_model_len", 8192)),
            "enforce_eager": bool(vllm_config.get("enforce_eager", False)),
        }
        optional_int_keys = ("max_num_seqs", "max_num_batched_tokens")
        for key in optional_int_keys:
            value = vllm_config.get(key)
            if value not in (None, 0):
                llm_kwargs[key] = int(value)
        self.llm = LLM(
            **llm_kwargs,
        )
        self.executor = VLLMSpectralExecutor(
            llm=self.llm,
            state=self.state,
            model_path=config["model"]["model_path"],
            max_new_tokens=int(config["generation"]["max_new_tokens"]),
            temperature=float(config["generation"].get("temperature", 0.0)),
            top_p=float(config["generation"].get("top_p", 1.0)),
            top_k=int(config["generation"].get("top_k", -1)),
            presence_penalty=float(config["generation"].get("presence_penalty", 0.0)),
            mutant_chunk_size=int(config["execution"]["mutant_chunk_size"]),
            adapter_root=self.adapter_root,
            rank=self.rank,
            reward_config={
                **config.get("reward", {}),
                "data_source": config.get("data", {}).get("source", "gsm8k"),
            },
        )

        if self.is_main_process:
            resolved_path = self.output_dir / "resolved_config.yaml"
            dump_yaml_config(resolved_path, self.config)
            self.log_event(
                "DATA_READY",
                {
                    "train_examples": len(self.train_records),
                    "val_examples": len(self.val_records),
                    "test_examples": len(self.test_records),
                    "svd_cache_path": str(self.svd_cache_path),
                    "parameterization": str(config.get("subspace", {}).get("parameterization", "spectral_dense")),
                    "spectral_rank": self.spectral_rank,
                    "adapter_rank": int(self.state.export_rank),
                    "world_size": self.world_size,
                    "mutants_per_worker": self.mutants_per_worker,
                },
            )
        print(
            "RANK_SHARD "
            + json.dumps(
                {
                    "rank": self.rank,
                    "world_size": self.world_size,
                    "local_rank": self.local_rank,
                    "local_mutant_start": self.local_shard.start,
                    "local_mutant_end": self.local_shard.end,
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
            flush=True,
        )
        dist.barrier()

    def _parameterization(self) -> str:
        return str(self.config.get("subspace", {}).get("parameterization", "spectral_dense")).strip().lower()

    def _resolved_svd_rank(self) -> int:
        if self._parameterization() == PARAMETERIZATION_FULL_FACTORIZED_M:
            return 0
        return int(self.config["subspace"]["rank"])

    def _ensure_svd_cache_ready_on_gpu(self) -> None:
        model_path = self.config["model"]["model_path"]
        svd_rank = self._resolved_svd_rank()
        band_strategy = self.config["subspace"]["band_strategy"]
        cache_dir = self.config["subspace"]["cache_dir"]
        svd_dtype = resolve_dtype(self.config["model"].get("svd_dtype", "float32"))
        preload_dtype = resolve_dtype(self.config["model"].get("dtype", "bfloat16"))

        builder_model = None
        cache_path: Path | None = None
        try:
            if self.rank == 0:
                device = (
                    torch.device(f"cuda:{self.local_rank}")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
                print(
                    "SVD_CACHE_PRECHECK "
                    + json.dumps(
                        {
                            "rank": self.rank,
                            "device": str(device),
                            "model_path": model_path,
                            "requested_rank": svd_rank,
                            "band_strategy": band_strategy,
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                    ),
                    flush=True,
                )
                builder_model = load_causal_lm(
                    model_path,
                    device=device,
                    dtype=preload_dtype,
                )
                selections = select_target_layers(
                    builder_model,
                    target_blocks=list(self.config["layers"]["target_blocks"]),
                    target_modules=list(self.config["layers"]["target_modules"]),
                )
                cache_path = resolve_svd_cache_path(
                    model_path=model_path,
                    selections=selections,
                    rank=svd_rank,
                    band_strategy=band_strategy,
                    cache_dir=cache_dir,
                )
                if cache_path.exists():
                    print(
                        f"SVD_CACHE_REUSE path={cache_path}",
                        flush=True,
                    )
                else:
                    print(
                        f"SVD_CACHE_BUILD_START path={cache_path} device={device}",
                        flush=True,
                    )
                    load_or_create_svd_cache(
                        model_path=model_path,
                        selections=selections,
                        rank=svd_rank,
                        band_strategy=band_strategy,
                        cache_dir=cache_dir,
                        device=device,
                        compute_dtype=svd_dtype,
                    )
                    print(
                        f"SVD_CACHE_BUILD_DONE path={cache_path}",
                        flush=True,
                    )
        finally:
            cleanup_cpu_model(builder_model)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        cache_path_str = str(cache_path) if cache_path is not None else None
        cache_path_str = self._broadcast_object(cache_path_str)
        dist.barrier()
        cache_path = Path(cache_path_str) if cache_path_str else None
        if cache_path is None or not cache_path.exists():
            raise RuntimeError("SVD cache was not materialized before normal trainer startup")

    def _init_process_group(self) -> None:
        if dist.is_initialized():
            return
        dist.init_process_group(backend="gloo", timeout=timedelta(minutes=30))

    def _validate_execution_config(self) -> None:
        execution_config = self.config.get("execution", {})
        distributed_mode = execution_config.get("distributed_mode", "single")
        if distributed_mode != "mutant_parallel":
            raise ValueError(f"expected execution.distributed_mode=mutant_parallel, got {distributed_mode}")
        es_config = self.config.get("es", {})
        antithetic = bool(es_config.get("antithetic", False))
        update_rule = es_config.get("update_rule", "per_layer_diagonal_cma_es")
        if antithetic and self.current_k % 2 != 0:
            raise ValueError("num_mutants must be even for antithetic sampling")
        if update_rule == "pairwise_directional" and not antithetic:
            raise ValueError("pairwise_directional requires es.antithetic=true")
        if update_rule not in {"pairwise_directional", "gaussian_mean", *CMA_UPDATE_RULES}:
            raise ValueError(f"unsupported es.update_rule: {update_rule}")
        if execution_config.get("backend", self.config.get("execution", {}).get("backend")) == "hf":
            raise ValueError("DistributedVLLMSpectralESTrainer only supports the vllm backend")
        resolve_mutant_shard(
            num_mutants=self.current_k,
            world_size=self.world_size,
            rank=self.rank,
            mutants_per_worker=self.mutants_per_worker,
        )

    def _init_wandb(self) -> None:
        if not self.is_main_process:
            return
        wandb_config = self.config.get("wandb", {})
        if not wandb_config.get("enabled", False):
            return
        wandb_dir = wandb_config.get("dir")
        if wandb_dir:
            Path(wandb_dir).mkdir(parents=True, exist_ok=True)
        mode = wandb_config.get("mode", "online")
        if mode == "online" and not os.environ.get("WANDB_API_KEY"):
            mode = "offline"
            print("WANDB_FALLBACK using offline mode because WANDB_API_KEY is not set", flush=True)
        tags = list(wandb_config.get("tags", []))
        if "vllm" not in tags:
            tags.append("vllm")
        if "mutant_parallel" not in tags:
            tags.append("mutant_parallel")
        self.wandb_run = wandb.init(
            project=wandb_config.get("project", "lowrank-spectral-es-rl"),
            entity=wandb_config.get("entity"),
            group=wandb_config.get("group") or self.config["algorithm"]["name"],
            job_type=wandb_config.get("job_type", "train"),
            name=wandb_config.get("name") or self.run_id,
            dir=wandb_dir,
            mode=mode,
            tags=tags,
            config=self.config,
            reinit="finish_previous",
        )
        self.wandb_run.define_metric("trainer/global_step")
        self.wandb_run.define_metric("*", step_metric="trainer/global_step")

    @staticmethod
    def _wandb_split_name(split: str) -> str:
        return "valid" if split == "val" else split

    @staticmethod
    def _merge_benchmark_metrics(gathered_payloads: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        merged: dict[str, dict[str, float]] = {}
        for item in gathered_payloads:
            for benchmark_name, stats in item.get("benchmark_metrics", {}).items():
                bucket = merged.setdefault(
                    benchmark_name,
                    {"num_examples": 0.0, "reward_sum": 0.0, "exact_match_sum": 0.0},
                )
                bucket["num_examples"] += float(stats.get("num_examples", 0.0))
                bucket["reward_sum"] += float(stats.get("reward_sum", 0.0))
                bucket["exact_match_sum"] += float(stats.get("exact_match_sum", 0.0))
        finalized: dict[str, dict[str, float]] = {}
        for benchmark_name in sorted(merged):
            num_examples = max(float(merged[benchmark_name]["num_examples"]), 1.0)
            finalized[benchmark_name] = {
                "num_examples": float(merged[benchmark_name]["num_examples"]),
                "accuracy": float(merged[benchmark_name]["exact_match_sum"]) / num_examples,
                "reward_mean": float(merged[benchmark_name]["reward_sum"]) / num_examples,
            }
        return finalized

    def _to_wandb_metrics(self, tag: str, payload: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
        if tag == "TRAIN_STEP":
            metrics = {f"train/{key}": value for key, value in payload.items() if isinstance(value, (int, float))}
            metrics["trainer/global_step"] = float(payload["step"])
            return metrics, {"trainer/tag": tag}
        if tag in {"VAL_ACCURACY", "EVAL_DONE"}:
            split = payload.get("split", "val")
            split_key = self._wandb_split_name(str(split))
            metrics = {
                f"{split_key}/accuracy": float(payload["accuracy"]),
                f"{split_key}/num_examples": float(payload["num_examples"]),
                f"{split_key}/reward_mean": float(payload.get("reward_mean", 0.0)),
                "trainer/global_step": float(payload["step"]),
            }
            for benchmark_name, stats in payload.get("benchmark_metrics", {}).items():
                metrics[f"{split_key}/{benchmark_name}/accuracy"] = float(stats["accuracy"])
                metrics[f"{split_key}/{benchmark_name}/num_examples"] = float(stats["num_examples"])
                metrics[f"{split_key}/{benchmark_name}/reward_mean"] = float(stats["reward_mean"])
            return metrics, {"trainer/tag": tag, "trainer/split": split}
        if tag == "BASELINE_DONE":
            split = payload.get("split", "baseline")
            split_key = self._wandb_split_name(str(split))
            metrics = {
                f"baseline/{split_key}/accuracy": float(payload["accuracy"]),
                f"baseline/{split_key}/num_examples": float(payload.get("num_examples", 0.0)),
                f"baseline/{split_key}/reward_mean": float(payload.get("reward_mean", 0.0)),
                "trainer/global_step": float(payload.get("step", 0.0)),
            }
            for benchmark_name, stats in payload.get("benchmark_metrics", {}).items():
                metrics[f"baseline/{split_key}/{benchmark_name}/accuracy"] = float(stats["accuracy"])
                metrics[f"baseline/{split_key}/{benchmark_name}/num_examples"] = float(stats["num_examples"])
                metrics[f"baseline/{split_key}/{benchmark_name}/reward_mean"] = float(stats["reward_mean"])
            return metrics, {"trainer/tag": tag}
        if tag == "DATA_READY":
            metrics = {
                **{f"data/{key}": float(value) for key, value in payload.items() if isinstance(value, (int, float))},
                "trainer/global_step": 0.0,
            }
            return metrics, {"trainer/tag": tag}
        if tag == "CHECKPOINT_SAVED":
            metrics = {
                "checkpoint/step": float(payload["step"]),
                "checkpoint/best_val_accuracy": float(payload["best_val_accuracy"]),
            }
            return metrics, {"trainer/tag": tag}
        return {}, {"trainer/tag": tag}

    @staticmethod
    def _now_beijing() -> datetime:
        return datetime.now(BEIJING_TZ)

    def log_event(self, tag: str, payload: dict[str, Any]) -> None:
        if not self.is_main_process:
            return
        now = self._now_beijing()
        record = {
            "tag": tag,
            "time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "timezone": "Asia/Shanghai",
            "timestamp": now.timestamp(),
            **payload,
        }
        console_payload = {
            "time": record["time"],
            "timezone": record["timezone"],
            **payload,
        }
        print(f"{tag} {json.dumps(console_payload, ensure_ascii=True, sort_keys=True)}", flush=True)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        if self.wandb_run is not None:
            metrics, extras = self._to_wandb_metrics(tag, payload)
            if extras:
                self.wandb_run.summary.update(extras)
            if metrics:
                self.wandb_run.log(metrics)

    def _select_records(self, split: str, max_examples: int = 0) -> list[dict]:
        mapping = {
            "train": self.train_records,
            "val": self.val_records,
            "test": self.test_records,
        }
        records = mapping[split]
        if max_examples and max_examples > 0:
            return records[: min(max_examples, len(records))]
        return records

    def _save_checkpoint(self, *, step: int, best_val_accuracy: float, name: str, current_k: int, current_micro_batch: int) -> Path:
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"
        payload = {
            "step": step,
            "best_val_accuracy": best_val_accuracy,
            "adapter_state": self.state.adapter_state_dict(),
            "cma_state": self.cma_state.state_dict() if self.cma_state is not None else None,
            "config": self.config,
            "current_k": current_k,
            "current_micro_batch": current_micro_batch,
        }
        torch.save(payload, checkpoint_path)
        self.log_event(
            "CHECKPOINT_SAVED",
            {
                "path": str(checkpoint_path),
                "step": step,
                "best_val_accuracy": best_val_accuracy,
            },
        )
        return checkpoint_path

    def _write_eval_output(self, *, output_dir: str | Path, predictions: list[dict], summary: dict[str, Any]) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if predictions:
            write_jsonl(output_dir / "predictions.jsonl", predictions)
            summary["predictions_path"] = str(output_dir / "predictions.jsonl")
        write_json(output_dir / "summary.json", summary)

    def _write_train_predictions(self, *, step: int, predictions: list[dict]) -> Path:
        step_dir = self.train_sample_root / f"step_{step:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        output_path = step_dir / f"rank_{self.rank:02d}.jsonl"
        write_jsonl(output_path, predictions[: self.TRAIN_SAMPLE_PREVIEW_LIMIT])
        return output_path

    def _resolve_eval_output_dir(self, *, split: str, step: int) -> Path:
        return self.eval_output_root / split / f"step_{step:04d}"

    def _resolve_eval_micro_batch(self) -> int:
        eval_config = self.config.get("eval", {})
        return int(eval_config.get("micro_batch", self.config["train"]["micro_batch"]))

    def _select_eval_records_for_rank(self, records: list[dict]) -> list[dict]:
        if self.world_size <= 1:
            return [
                {**record, "_record_index": index}
                for index, record in enumerate(records)
            ]
        num_records = len(records)
        base = num_records // self.world_size
        remainder = num_records % self.world_size
        start = self.rank * base + min(self.rank, remainder)
        count = base + (1 if self.rank < remainder else 0)
        end = start + count
        return [
            {**records[index], "_record_index": index}
            for index in range(start, end)
        ]

    def _aggregate_eval_payloads(
        self,
        *,
        gathered_payloads: list[dict[str, Any]],
        split: str,
        step: int,
        output_dir: str | Path | None,
    ) -> dict:
        total_examples = int(sum(int(item["num_examples"]) for item in gathered_payloads))
        reward_sum = sum(float(item["reward_sum"]) for item in gathered_payloads)
        exact_match_sum = sum(float(item["exact_match_sum"]) for item in gathered_payloads)

        elapsed_seconds = max(float(item["profiler_snapshot"].get("elapsed_seconds", 0.0)) for item in gathered_payloads)
        elapsed_seconds = max(elapsed_seconds, 1e-6)
        generated_tokens_total = sum(float(item["profiler_snapshot"].get("generated_tokens_total", 0.0)) for item in gathered_payloads)
        requests_total = sum(float(item["profiler_snapshot"].get("requests_total", 0.0)) for item in gathered_payloads)
        mutant_evals_total = sum(float(item["profiler_snapshot"].get("mutant_evals_total", 0.0)) for item in gathered_payloads)

        total_examples = max(total_examples, 1)
        benchmark_metrics = self._merge_benchmark_metrics(gathered_payloads)
        summary = {
            "num_examples": total_examples,
            "accuracy": exact_match_sum / total_examples,
            "reward_mean": reward_sum / total_examples,
            "benchmark_metrics": benchmark_metrics,
            "profiler": {
                "elapsed_seconds": elapsed_seconds,
                "generated_tokens_total": generated_tokens_total,
                "requests_total": requests_total,
                "mutant_evals_total": mutant_evals_total,
                "tokens_per_sec": generated_tokens_total / elapsed_seconds,
                "requests_per_sec": requests_total / elapsed_seconds,
                "mutants_per_sec": mutant_evals_total / elapsed_seconds,
            },
        }

        if output_dir is not None:
            merged_predictions: list[dict[str, Any]] = []
            for item in gathered_payloads:
                merged_predictions.extend(item.get("predictions", []))
            merged_predictions.sort(key=lambda item: int(item.get("record_index", 0)))
            for prediction in merged_predictions:
                prediction.pop("record_index", None)
            self._write_eval_output(output_dir=output_dir, predictions=merged_predictions, summary=summary)

        self.log_event(
            "VAL_ACCURACY" if split == "val" else "EVAL_DONE",
            {
                "split": split,
                "step": step,
                "accuracy": summary["accuracy"],
                "reward_mean": summary["reward_mean"],
                "num_examples": summary["num_examples"],
                "benchmark_metrics": benchmark_metrics,
            },
        )
        return summary

    def _evaluate_split(self, *, split: str, step: int, max_examples: int = 0, output_dir: str | Path | None = None) -> dict:
        records = self._select_records(split, max_examples=max_examples)
        if self.is_main_process:
            self.log_event("EVAL_START", {"split": split, "step": step, "num_examples": len(records)})
        self.state.activate_current_state()
        local_records = self._select_eval_records_for_rank(records)
        result = self.executor.score_current_state(
            records=local_records,
            question_micro_batch=min(self._resolve_eval_micro_batch(), max(1, len(local_records))),
            collect_predictions=output_dir is not None,
            use_base_model=step == 0,
        )
        local_count = len(local_records)
        local_payload = {
            "rank": self.rank,
            "num_examples": local_count,
            "reward_sum": float(result.rewards[0].item()) * local_count if result.rewards.numel() else 0.0,
            "exact_match_sum": float(result.exact_match_rates[0].item()) * local_count if result.exact_match_rates.numel() else 0.0,
            "profiler_snapshot": result.profiler_snapshot,
            "predictions": result.predictions,
            "benchmark_metrics": result.benchmark_metrics,
        }
        gathered_payloads: list[dict[str, Any] | None] = [None] * self.world_size
        dist.all_gather_object(gathered_payloads, local_payload)
        if not self.is_main_process:
            return {}
        return self._aggregate_eval_payloads(
            gathered_payloads=[item for item in gathered_payloads if item is not None],
            split=split,
            step=step,
            output_dir=output_dir,
        )

    def run_baseline(self, *, split: str = "test", max_examples: int = 0) -> dict:
        if self.is_main_process:
            output_root = self._resolve_eval_output_dir(split=split, step=0)
            self.log_event(
                "BASELINE_START",
                {"split": split, "max_examples": max_examples if max_examples > 0 else len(self._select_records(split))},
            )
            summary = self._evaluate_split(split=split, step=0, max_examples=max_examples, output_dir=output_root)
            self.log_event(
                "BASELINE_DONE",
                {
                    "split": split,
                    "step": 0,
                    "accuracy": summary["accuracy"],
                    "reward_mean": summary["reward_mean"],
                    "num_examples": summary["num_examples"],
                    "benchmark_metrics": summary.get("benchmark_metrics", {}),
                },
            )
        else:
            summary = {}
        dist.barrier()
        return summary

    def _broadcast_object(self, payload: Any) -> Any:
        items = [payload]
        dist.broadcast_object_list(items, src=0)
        return items[0]

    def _sample_train_batch(self, batch_size: int) -> list[dict]:
        if self.is_main_process:
            if not self.train_records:
                raise ValueError("training split is empty; check data split configuration")
            if batch_size <= len(self.train_records):
                batch_indices = random.sample(range(len(self.train_records)), batch_size)
            else:
                batch_indices = random.choices(range(len(self.train_records)), k=batch_size)
        else:
            batch_indices = None
        batch_indices = self._broadcast_object(batch_indices)
        return [self.train_records[index] for index in batch_indices]

    def _sample_noise(self, current_k: int) -> dict[str, dict[str, torch.Tensor]]:
        if self.is_main_process:
            antithetic = bool(self.config.get("es", {}).get("antithetic", False))
            update_rule = self.config.get("es", {}).get("update_rule", "per_layer_diagonal_cma_es")
            if update_rule in CMA_UPDATE_RULES:
                if self.cma_state is None:
                    raise RuntimeError(f"{update_rule} requires CMA state to be initialized")
                noise_payloads = self.cma_state.sample_noise(current_k, antithetic=antithetic)
            else:
                noise_payloads = self.state.sample_noise(current_k, antithetic=antithetic)
        else:
            noise_payloads = None
        return self._broadcast_object(noise_payloads)

    def _collect_rank_payload(
        self,
        *,
        local_rewards: torch.Tensor,
        local_exact_match_rates: torch.Tensor,
        profiler_snapshot: dict[str, float],
        gpu_snapshot: GPUMonitorSnapshot,
        step_duration_seconds: float,
        local_shard: MutantShard,
        train_samples_path: Path,
    ) -> list[dict[str, Any]]:
        local_payload = {
            "rank": self.rank,
            "local_mutant_start": local_shard.start,
            "local_mutant_end": local_shard.end,
            "rewards": [float(item) for item in local_rewards.detach().cpu().tolist()],
            "exact_match_rates": [float(item) for item in local_exact_match_rates.detach().cpu().tolist()],
            "profiler_snapshot": profiler_snapshot,
            "gpu_snapshot": asdict(gpu_snapshot),
            "step_duration_seconds": float(step_duration_seconds),
            "train_samples_path": str(train_samples_path),
        }
        gathered_payloads: list[dict[str, Any] | None] = [None] * self.world_size
        dist.all_gather_object(gathered_payloads, local_payload)
        return [item for item in gathered_payloads if item is not None]

    def _train_one_step(self, *, step: int, current_k: int, current_micro_batch: int) -> dict[str, Any]:
        batch_records = self._sample_train_batch(int(self.config["train"]["effective_question_batch"]))
        if self.is_main_process:
            self.log_event(
                "TRAIN_LOOP_START",
                {
                    "step": step,
                    "current_k": current_k,
                    "current_micro_batch": current_micro_batch,
                    "effective_question_batch": int(self.config["train"]["effective_question_batch"]),
                    "world_size": self.world_size,
                    "mutants_per_worker": self.mutants_per_worker,
                },
            )
        noise_payloads = self._sample_noise(current_k)
        self.state.activate_mutants(noise_payloads, self.config["es"]["sigma"])

        started_at = time.time()
        self.gpu_monitor.start()
        step_result = self.executor.score_mutant_subset(
            records=batch_records,
            mutant_indices=self.local_shard.indices,
            question_micro_batch=current_micro_batch,
            collect_predictions=True,
        )
        gpu_stats = self.gpu_monitor.stop()
        step_duration_seconds = time.time() - started_at
        local_train_samples_path = self._write_train_predictions(step=step, predictions=step_result.predictions)

        rank_payloads = self._collect_rank_payload(
            local_rewards=step_result.rewards,
            local_exact_match_rates=step_result.exact_match_rates,
            profiler_snapshot=step_result.profiler_snapshot,
            gpu_snapshot=gpu_stats,
            step_duration_seconds=step_duration_seconds,
            local_shard=self.local_shard,
            train_samples_path=local_train_samples_path,
        )

        if self.is_main_process:
            ordered_payloads = sorted(rank_payloads, key=lambda item: int(item["rank"]))
            ordered_shards = [
                MutantShard(
                    rank=int(item["rank"]),
                    world_size=self.world_size,
                    start=int(item["local_mutant_start"]),
                    end=int(item["local_mutant_end"]),
                )
                for item in ordered_payloads
            ]
            global_rewards = merge_sharded_rewards(
                num_mutants=current_k,
                shards=ordered_shards,
                shard_rewards=[list(item["rewards"]) for item in ordered_payloads],
            )
            global_exact_match_rates = merge_sharded_rewards(
                num_mutants=current_k,
                shards=ordered_shards,
                shard_rewards=[list(item["exact_match_rates"]) for item in ordered_payloads],
            )
            update_rule = self.config.get("es", {}).get("update_rule", "per_layer_diagonal_cma_es")
            if update_rule == "pairwise_directional":
                direction_payloads, direction_stats = compute_pairwise_direction_payloads(
                    noise_payloads=noise_payloads,
                    rewards=global_rewards,
                )
            elif update_rule == "gaussian_mean":
                direction_payloads, direction_stats = compute_gaussian_direction_payloads(
                    noise_payloads=noise_payloads,
                    rewards=global_rewards,
                )
                trust_region = self.config.get("es", {}).get("trust_region", {})
                step_payloads, step_stats = apply_alpha_update_to_direction_payloads(
                    direction_payloads=direction_payloads,
                    alpha_config=self.config["es"].get(
                        "alpha",
                        self.config["es"].get("learning_rate", self.config["es"].get("step_size", {"m": 0.005})),
                    ),
                    sigma_config=self.config["es"]["sigma"],
                    max_layer_step_config=trust_region.get("max_layer_step_norm"),
                )
            elif update_rule in CMA_UPDATE_RULES:
                if self.cma_state is None:
                    raise RuntimeError(f"{update_rule} requires CMA state to be initialized")
                trust_region = self.config.get("es", {}).get("trust_region", {})
                max_state_norm = None
                if "max_state_norm" in trust_region:
                    max_state_norm = resolve_named_value(trust_region["max_state_norm"], "m")
                step_payloads, direction_stats, step_stats = self.cma_state.apply_update(
                    rewards=global_rewards,
                    noise_payloads=noise_payloads,
                    current_states={name: adapter.m_state.detach().cpu().clone() for name, adapter in self.state.adapters.items()},
                    max_layer_step_config=trust_region.get("max_layer_step_norm"),
                    max_state_norm=max_state_norm,
                )
            else:
                raise ValueError(f"unsupported es.update_rule: {update_rule}")
            if update_rule == "pairwise_directional":
                trust_region = self.config.get("es", {}).get("trust_region", {})
                step_payloads, step_stats = apply_alpha_update_to_direction_payloads(
                    direction_payloads=direction_payloads,
                    alpha_config=self.config["es"].get(
                        "alpha",
                        self.config["es"].get("learning_rate", self.config["es"].get("step_size", {"m": 0.005})),
                    ),
                    sigma_config=self.config["es"]["sigma"],
                    max_layer_step_config=trust_region.get("max_layer_step_norm"),
                )
        else:
            global_rewards = None
            global_exact_match_rates = None
            direction_stats = None
            step_stats = None
            step_payloads = None

        step_payloads = self._broadcast_object(step_payloads)
        trust_region = self.config.get("es", {}).get("trust_region", {})
        max_state_norm = None
        if "max_state_norm" in trust_region:
            max_state_norm = resolve_named_value(trust_region["max_state_norm"], "m")
        self.state.apply_step_payloads(step_payloads, max_state_norm=max_state_norm)
        self.state.activate_current_state()

        if not self.is_main_process or global_rewards is None:
            return {}

        aggregated_step_metrics = aggregate_distributed_step_metrics(
            profiler_snapshots=[dict(item["profiler_snapshot"]) for item in ordered_payloads],
            gpu_snapshots=[GPUMonitorSnapshot(**item["gpu_snapshot"]) for item in ordered_payloads],
            step_durations=[float(item["step_duration_seconds"]) for item in ordered_payloads],
        )
        adapter_norms = self.state.adapter_norms()
        payload = {
            "step": step,
            "reward_mean": float(global_rewards.mean().item()),
            "reward_std": float(global_rewards.std(unbiased=False).item()),
            "accuracy_mean": float(global_exact_match_rates.mean().item()),
            "current_k": current_k,
            "current_micro_batch": current_micro_batch,
            "peak_memory_gb": 0.0,
            "adapter_norm_sum": float(sum(adapter_norms.values())),
            "update_accepted": 1.0,
            "step_scale": 1.0,
            "world_size": self.world_size,
            "mutants_per_worker": self.mutants_per_worker,
            "local_mutant_start": self.local_shard.start,
            "local_mutant_end": self.local_shard.end,
            "shard_ranges": [[item.start, item.end] for item in ordered_shards],
            "train_samples_dir": str(self.train_sample_root / f"step_{step:04d}"),
            **(direction_stats or {}),
            **(step_stats or {}),
            **aggregated_step_metrics,
        }
        self.log_event("TRAIN_STEP", payload)
        return payload

    def run(self) -> dict:
        eval_split = str(self.config.get("eval", {}).get("split", "val"))
        if eval_split not in {"val", "test"}:
            raise ValueError(f"unsupported eval.split: {eval_split}")
        if self.config.get("eval", {}).get("skip_initial_validation", False):
            best_val_accuracy = float("-inf")
            if self.is_main_process:
                self.log_event("INITIAL_VAL_SKIPPED", {"reason": "config.eval.skip_initial_validation"})
        else:
            best_summary = self._evaluate_split(
                split=eval_split,
                step=0,
                output_dir=self._resolve_eval_output_dir(split=eval_split, step=0),
            )
            if self.is_main_process:
                self.log_event(
                    "BASELINE_DONE",
                    {
                        "split": eval_split,
                        "step": 0,
                        "accuracy": best_summary["accuracy"],
                        "reward_mean": best_summary["reward_mean"],
                        "num_examples": best_summary["num_examples"],
                        "benchmark_metrics": best_summary.get("benchmark_metrics", {}),
                    },
                )
                best_val_accuracy = best_summary["accuracy"]
            else:
                best_val_accuracy = float("-inf")
            dist.barrier()

        current_micro_batch = int(self.config["train"]["micro_batch"])
        current_k = int(self.config["es"]["num_mutants"])

        for step in range(1, int(self.config["train"]["train_steps"]) + 1):
            self._train_one_step(step=step, current_k=current_k, current_micro_batch=current_micro_batch)
            if self.is_main_process:
                self._save_checkpoint(
                    step=step,
                    best_val_accuracy=best_val_accuracy,
                    name=f"step_{step:04d}",
                    current_k=current_k,
                    current_micro_batch=current_micro_batch,
                )
            if step % int(self.config["eval"]["eval_every_steps"]) == 0 or step == int(self.config["train"]["train_steps"]):
                dist.barrier()
                summary = self._evaluate_split(
                    split=eval_split,
                    step=step,
                    output_dir=self._resolve_eval_output_dir(split=eval_split, step=step),
                )
                if self.is_main_process:
                    self._save_checkpoint(
                        step=step,
                        best_val_accuracy=best_val_accuracy,
                        name="last",
                        current_k=current_k,
                        current_micro_batch=current_micro_batch,
                    )
                    if summary["accuracy"] >= best_val_accuracy:
                        best_val_accuracy = summary["accuracy"]
                        self._save_checkpoint(
                            step=step,
                            best_val_accuracy=best_val_accuracy,
                            name="best",
                            current_k=current_k,
                            current_micro_batch=current_micro_batch,
                        )
                dist.barrier()

        final_summary = {
            "best_val_accuracy": best_val_accuracy,
            "current_k": current_k,
            "current_micro_batch": current_micro_batch,
            "world_size": self.world_size,
            "mutants_per_worker": self.mutants_per_worker,
            "output_dir": str(self.output_dir),
        }
        if self.is_main_process:
            write_json(self.output_dir / "run_summary.json", final_summary)
            if self.wandb_run is not None:
                self.wandb_run.summary.update(final_summary)
        dist.barrier()
        return final_summary

    def close(self) -> None:
        if self.wandb_run is not None:
            self.wandb_run.finish()
            self.wandb_run = None
        llm = getattr(self, "llm", None)
        if llm is not None:
            del self.llm
        if dist.is_initialized():
            dist.destroy_process_group()
