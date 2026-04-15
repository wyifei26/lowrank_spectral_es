#!/usr/bin/env python3

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml


ROOT = Path("/GenSIvePFS/users/yfwang")
PROJECT_DIR = ROOT / "lowrank_spectral_es_rl"
OUTPUT_ROOT_DIR = PROJECT_DIR / "runs"
WANDB_DIR = ROOT / "output" / "wandb"
SVD_CACHE_DIR = PROJECT_DIR / "artifacts" / "svd_cache"
VOLC_CONFIG_ROOT = ROOT / "volc" / "task-configs"
DATASETS_DIR = PROJECT_DIR / "dataset"
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
GPU_FLAVORS = {
    1: "ml.pni2.3xlarge",
    2: "ml.pni2.7xlarge",
    4: "ml.pni2.14xlarge",
    8: "ml.pni2.28xlarge",
}
BEIJING_TZ = ZoneInfo("Asia/Shanghai")


@dataclass(frozen=True)
class DatasetPreset:
    key: str
    task_config_subdir: str
    base_config_path: str
    max_new_tokens: int
    data_overrides: dict[str, Any]
    prompt_overrides: dict[str, Any]
    generation_overrides: dict[str, Any]
    vllm_overrides: dict[str, Any]
    tags: tuple[str, ...]
    description_label: str


@dataclass(frozen=True)
class ModelPreset:
    key: str
    model_path: str
    thinking: bool
    num_mutants: int
    effective_question_batch: int
    eval_micro_batch: int
    max_cpu_loras: int
    gpu_memory_utilization: float = 0.85
    model_dtype: str = "bfloat16"
    svd_dtype: str = "float32"


@dataclass(frozen=True)
class ParamPreset:
    key: str
    train_steps: int
    eval_every_steps: int
    eval_split: str
    skip_initial_validation: bool
    target_blocks: str | int | list[int] = "all-blocks"
    target_modules: tuple[str, ...] = tuple(DEFAULT_TARGET_MODULES)
    band_strategy: str = "top-band"
    reward_exact_match: float = 1.0
    split_seed: int = 42
    seed: int = 42
    trust_region_max_layer_step_norm_m: float = 0.0
    trust_region_max_state_norm_m: float = 0.0
    cma_selection_ratio: float = 0.75
    cma_mean_step_scale: float = 1.0
    cma_min_sigma: float = 1.0e-6
    cma_max_sigma: float = 0.02
    cma_min_eigenvalue: float = 1.0e-8
    wandb_project: str = "lowrank-spectral-es-rl"
    notes_tag: str | None = None
    wandb_mode: str = "online"
    wandb_enabled: bool = True


@dataclass(frozen=True)
class ParameterizationPreset:
    key: str
    parameterization: str
    sigma_m: float
    subspace_rank: int | None = None
    factor_rank: int = 0
    factor_init_scale: float = 0.001
    label: str = ""
    summary: str = ""


@dataclass(frozen=True)
class SamplingPreset:
    key: str
    update_rule: str
    antithetic: bool
    alpha_m: float | None = None
    mechanism_label: str = ""
    mechanism_summary: str = ""


@dataclass(frozen=True)
class TaskSelection:
    dataset: DatasetPreset
    model: ModelPreset
    parameterization: ParameterizationPreset
    params: ParamPreset
    sampling: SamplingPreset
    gpu_count: int
    wandb_group: str
    output_root_dir: str = str(OUTPUT_ROOT_DIR)


@dataclass(frozen=True)
class ResolvedTaskConfig:
    dataset: DatasetPreset
    model: ModelPreset
    parameterization: ParameterizationPreset
    params: ParamPreset
    sampling: SamplingPreset
    gpu_count: int
    world_size: int
    gpus_per_node: int
    flavor: str
    mutants_per_worker: int
    mutant_chunk_size: int
    train_micro_batch: int
    output_root_dir: str
    wandb_group: str


DATASET_PRESETS: dict[str, DatasetPreset] = {
    "math_data": DatasetPreset(
        key="math_data",
        task_config_subdir="math_data",
        base_config_path="configs/spectral_es_vllm_mutant_parallel.yaml",
        max_new_tokens=16000,
        data_overrides={
            "source": "math_data",
            "root_dir": str(DATASETS_DIR / "math_data"),
            "split_seed": 42,
            "val_size": 748,
        },
        prompt_overrides={
            "template_name": "math_boxed_v1",
            "require_box_answer": True,
            "use_chat_template": True,
        },
        generation_overrides={
            "temperature": 0.0,
        },
        vllm_overrides={
            "enforce_eager": True,
        },
        tags=("math_data",),
        description_label="Math Data",
    ),
    "gsm8k": DatasetPreset(
        key="gsm8k",
        task_config_subdir="gsm8k",
        base_config_path="configs/spectral_es_vllm_mutant_parallel.yaml",
        max_new_tokens=2048,
        data_overrides={
            "source": "gsm8k",
            "root_dir": str(DATASETS_DIR / "gsm8k"),
            "split_seed": 42,
            "val_size": 748,
        },
        prompt_overrides={
            "template_name": "math_boxed_v1",
            "require_box_answer": True,
            "use_chat_template": True,
        },
        generation_overrides={
            "temperature": 0.0,
        },
        vllm_overrides={
            "enforce_eager": True,
        },
        tags=("gsm8k",),
        description_label="GSM8K",
    ),
    "mmlu_pro": DatasetPreset(
        key="mmlu_pro",
        task_config_subdir="mmlu_pro",
        base_config_path="configs/spectral_es_vllm_mutant_parallel_mmlu_pro.yaml",
        max_new_tokens=4096,
        data_overrides={
            "source": "mmlu_pro",
            "root_dir": str(DATASETS_DIR / "mmlu_pro"),
            "raw_splits": ["validation", "test"],
            "split_seed": 42,
            "train_ratio": 0.94,
            "val_ratio": 0.03,
            "test_ratio": 0.03,
            "val_size": 0,
        },
        prompt_overrides={
            "template_name": "mmlu_pro_boxed_choice_v1",
            "require_box_answer": True,
            "use_chat_template": True,
        },
        generation_overrides={
            "temperature": 0.0,
        },
        vllm_overrides={
            "enforce_eager": True,
        },
        tags=("mmlu_pro",),
        description_label="MMLU-Pro",
    ),
}


MODEL_PRESETS: dict[str, ModelPreset] = {
    "qwen3_0p6b_base": ModelPreset(
        key="qwen3_0p6b_base",
        model_path="/GenSIvePFS/users/model/Qwen/Qwen3-0.6B-Base",
        thinking=False,
        num_mutants=64,
        effective_question_batch=64,
        eval_micro_batch=512,
        max_cpu_loras=32,
    ),
    "qwen3_0p6b_nothink": ModelPreset(
        key="qwen3_0p6b_nothink",
        model_path="/GenSIvePFS/users/model/Qwen/Qwen3-0.6B",
        thinking=False,
        num_mutants=32,
        effective_question_batch=128,
        eval_micro_batch=512,
        max_cpu_loras=32,
    ),
    "qwen3_0p6b_think": ModelPreset(
        key="qwen3_0p6b_think",
        model_path="/GenSIvePFS/users/model/Qwen/Qwen3-0.6B",
        thinking=True,
        num_mutants=64,
        effective_question_batch=64,
        eval_micro_batch=512,
        max_cpu_loras=32,
    ),
    "qwen3_1p7b_base": ModelPreset(
        key="qwen3_1p7b_base",
        model_path="/GenSIvePFS/users/model/Qwen/Qwen3-1.7B-Base",
        thinking=False,
        num_mutants=64,
        effective_question_batch=128,
        eval_micro_batch=512,
        max_cpu_loras=64,
    ),
    "qwen3_1p7b_base_rl": ModelPreset(
        key="qwen3_1p7b_base_rl",
        model_path="/GenSIvePFS/users/yfwang/code/verl/trajectories/verl_checkpoints/rl_recipe_qwen3_1p7b_base_gsm8k_4gpu_maxtok4096_temp0p6_topp0p9_e3_norsm_qwen3_1p7b_base_hf/global_step_65/actor/huggingface",
        thinking=False,
        num_mutants=64,
        effective_question_batch=128,
        eval_micro_batch=512,
        max_cpu_loras=64,
    ),
    "qwen3_8b_base": ModelPreset(
        key="qwen3_8b_base",
        model_path="/GenSIvePFS/users/model/Qwen/Qwen3-8B-Base",
        thinking=True,
        num_mutants=64,
        effective_question_batch=64,
        eval_micro_batch=256,
        max_cpu_loras=64,
    ),
}


SAMPLING_PRESETS: dict[str, SamplingPreset] = {
    # 1) Mirrored ES with antithetic pairwise reward differences.
    "pairwise": SamplingPreset(
        key="pairwise",
        update_rule="pairwise_directional",
        antithetic=True,
        alpha_m=0.005,
        mechanism_label="Pairwise directional ES",
        mechanism_summary="antithetic mirrored ES with pairwise reward differences",
    ),
    # 2) Plain Gaussian ES with reward-weighted mean noise directions.
    "gaussian": SamplingPreset(
        key="gaussian",
        update_rule="gaussian_mean",
        antithetic=False,
        alpha_m=0.005,
        mechanism_label="Gaussian mean ES",
        mechanism_summary="standard Gaussian ES using reward-weighted mean directions",
    ),
    # 3) Per-layer diagonal CMA-ES with layerwise variance and sigma adaptation.
    "cma": SamplingPreset(
        key="cma",
        update_rule="per_layer_diagonal_cma_es",
        antithetic=False,
        alpha_m=None,
        mechanism_label="Per-layer Diagonal CMA-ES",
        mechanism_summary="layerwise diagonal covariance adaptation with optional antithetic latent sampling",
    ),
}


PARAMETERIZATION_PRESETS: dict[str, ParameterizationPreset] = {
    "spectral_dense": ParameterizationPreset(
        key="spectral_dense",
        parameterization="spectral_dense",
        sigma_m=0.02,
        subspace_rank=32,
        label="Spectral dense M",
        summary="truncated spectral basis with a dense in-basis matrix M",
    ),
    "lora_es": ParameterizationPreset(
        key="lora_es",
        parameterization="lora_es",
        sigma_m=0.004,
        factor_rank=8,
        label="LoRA-ES",
        summary="direct low-rank LoRA perturbation in weight space",
    ),
    "full_factorized_m": ParameterizationPreset(
        key="full_factorized_m",
        parameterization="full_factorized_m",
        sigma_m=0.005,
        factor_rank=8,
        label="Full-basis factorized M",
        summary="full spectral basis with factorized in-basis matrix M = P Q^T",
    ),
}


PARAM_PRESETS: dict[str, ParamPreset] = {
    "default": ParamPreset(
        key="default",
        train_steps=300,
        eval_every_steps=5,
        eval_split="test",
        skip_initial_validation=False,
        notes_tag=None,
    ),
}


def resolve_task_selection(selection: TaskSelection) -> ResolvedTaskConfig:
    gpu_count = selection.gpu_count
    if gpu_count not in GPU_FLAVORS:
        raise ValueError(f"gpu_count must be one of {sorted(GPU_FLAVORS)}, got {gpu_count}")
    num_mutants = selection.model.num_mutants
    if num_mutants % gpu_count != 0:
        raise ValueError(
            f"num_mutants must be divisible by gpu_count, got {num_mutants} vs {gpu_count}"
        )
    mutants_per_worker = num_mutants // gpu_count
    return ResolvedTaskConfig(
        dataset=selection.dataset,
        model=selection.model,
        parameterization=selection.parameterization,
        params=selection.params,
        sampling=selection.sampling,
        gpu_count=gpu_count,
        world_size=gpu_count,
        gpus_per_node=gpu_count,
        flavor=GPU_FLAVORS[gpu_count],
        mutants_per_worker=mutants_per_worker,
        mutant_chunk_size=mutants_per_worker,
        train_micro_batch=selection.model.effective_question_batch,
        output_root_dir=selection.output_root_dir,
        wandb_group=selection.wandb_group,
    )


def timestamp_now() -> str:
    return datetime.now(BEIJING_TZ).strftime("%m%d-%H%M")


def format_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ",".join(format_override_value(item) for item in value) + "]"
    return str(value)


def target_blocks_tag(value: str | int | list[int]) -> str:
    if value == "all-blocks":
        return "allblocks"
    if isinstance(value, int):
        return f"first{value}blocks"
    if isinstance(value, list):
        return f"custom{len(value)}blocks"
    raise TypeError(f"unsupported target_blocks value: {type(value)}")


def basis_rank_tag(parameterization: ParameterizationPreset) -> str:
    if parameterization.parameterization == "full_factorized_m":
        return "fullbasis"
    if parameterization.subspace_rank is None:
        return ""
    return f"r{parameterization.subspace_rank}"


def factor_rank_tag(parameterization: ParameterizationPreset) -> str | None:
    if parameterization.factor_rank <= 0:
        return None
    return f"fr{parameterization.factor_rank}"


def build_run_suffix(config: ResolvedTaskConfig, timestamp: str) -> str:
    model_tag = Path(config.model.model_path).name.lower().replace(".", "p")
    thinking_tag = "thinking" if config.model.thinking else "nothinking"
    dataset_tag = config.dataset.key
    blocks_tag = target_blocks_tag(config.params.target_blocks)
    sampling_tag = config.sampling.key.replace("_default", "")
    suffix_parts = [
        timestamp,
        dataset_tag,
        sampling_tag,
        blocks_tag,
        "allmodules",
        config.parameterization.key,
        f"k{config.model.num_mutants}",
        f"q{config.model.effective_question_batch}",
        f"mb{config.train_micro_batch}",
        f"chunk{config.mutant_chunk_size}",
        f"{config.gpu_count}gpu",
        f"sigma{str(config.parameterization.sigma_m).replace('.', 'p')}",
        model_tag,
        thinking_tag,
    ]
    basis_tag = basis_rank_tag(config.parameterization)
    if basis_tag:
        suffix_parts.append(basis_tag)
    factor_tag = factor_rank_tag(config.parameterization)
    if factor_tag:
        suffix_parts.append(factor_tag)
    if config.params.notes_tag:
        suffix_parts.append(config.params.notes_tag)
    return "_".join(suffix_parts)


def build_overrides(config: ResolvedTaskConfig, run_id: str) -> list[str]:
    generation_overrides = {
        **config.dataset.generation_overrides,
        "max_new_tokens": config.dataset.max_new_tokens,
    }
    vllm_overrides = {
        "max_model_len": config.dataset.max_new_tokens,
        **config.dataset.vllm_overrides,
    }
    overrides = [
        f"seed={config.params.seed}",
        f"model.model_path={config.model.model_path}",
        f"model.dtype={config.model.model_dtype}",
        f"model.svd_dtype={config.model.svd_dtype}",
        "execution.distributed_mode=mutant_parallel",
        "execution.backend=vllm",
        f"execution.world_size={config.world_size}",
        f"execution.gpus_per_node={config.gpus_per_node}",
        f"execution.mutants_per_worker={config.mutants_per_worker}",
        f"execution.mutant_chunk_size={config.mutant_chunk_size}",
        f"es.num_mutants={config.model.num_mutants}",
        f"es.update_rule={config.sampling.update_rule}",
        f"es.antithetic={format_override_value(config.sampling.antithetic)}",
        f"es.sigma.m={config.parameterization.sigma_m}",
        f"es.cma.selection_ratio={config.params.cma_selection_ratio}",
        f"es.cma.mean_step_scale={config.params.cma_mean_step_scale}",
        f"es.cma.min_sigma={config.params.cma_min_sigma}",
        f"es.cma.max_sigma={config.params.cma_max_sigma}",
        f"es.cma.min_eigenvalue={config.params.cma_min_eigenvalue}",
        f"es.trust_region.max_layer_step_norm.m={config.params.trust_region_max_layer_step_norm_m}",
        f"es.trust_region.max_state_norm.m={config.params.trust_region_max_state_norm_m}",
        f"reward.exact_match={config.params.reward_exact_match}",
        f"train.train_steps={config.params.train_steps}",
        f"train.effective_question_batch={config.model.effective_question_batch}",
        f"train.micro_batch={config.train_micro_batch}",
        f"eval.micro_batch={config.model.eval_micro_batch}",
        f"eval.eval_every_steps={config.params.eval_every_steps}",
        f"eval.skip_initial_validation={format_override_value(config.params.skip_initial_validation)}",
        f"eval.split={config.params.eval_split}",
        f"subspace.parameterization={config.parameterization.parameterization}",
        f"subspace.factor_rank={config.parameterization.factor_rank}",
        f"subspace.factor_init_scale={config.parameterization.factor_init_scale}",
        f"subspace.band_strategy={config.params.band_strategy}",
        f"subspace.cache_dir={SVD_CACHE_DIR}",
        f"vllm.max_cpu_loras={config.model.max_cpu_loras}",
        f"vllm.gpu_memory_utilization={config.model.gpu_memory_utilization}",
        f"layers.target_blocks={format_override_value(config.params.target_blocks)}",
        f"layers.target_modules={format_override_value(list(config.params.target_modules))}",
        f"output.root_dir={config.output_root_dir}",
        f"output.run_id={run_id}",
        f"wandb.enabled={format_override_value(config.params.wandb_enabled)}",
        f"wandb.project={config.params.wandb_project}",
        "wandb.entity=null",
        f"wandb.group={config.wandb_group}",
        "wandb.job_type=train",
        f"wandb.name={run_id}",
        f"wandb.dir={WANDB_DIR}",
        f"wandb.mode={config.params.wandb_mode}",
    ]
    if config.sampling.alpha_m is not None:
        overrides.append(f"es.alpha.m={config.sampling.alpha_m}")
    if config.parameterization.subspace_rank is not None:
        overrides.append(f"subspace.rank={config.parameterization.subspace_rank}")
    overrides.extend(_dict_to_overrides("data", config.dataset.data_overrides))
    overrides.extend(
        _dict_to_overrides(
            "prompt",
            {**config.dataset.prompt_overrides, "enable_thinking": config.model.thinking},
        )
    )
    overrides.extend(_dict_to_overrides("generation", generation_overrides))
    overrides.extend(_dict_to_overrides("vllm", vllm_overrides))
    model_name = Path(config.model.model_path).name
    tags = list(config.dataset.tags) + [
        "spectral_es",
        "vllm",
        "mutant_parallel",
        model_name,
        f"{config.params.eval_split}_eval",
        config.parameterization.key,
    ]
    basis_tag = basis_rank_tag(config.parameterization)
    if basis_tag:
        tags.append(basis_tag)
    factor_tag = factor_rank_tag(config.parameterization)
    if factor_tag:
        tags.append(factor_tag)
    if config.params.notes_tag:
        tags.append(config.params.notes_tag)
    overrides.append(f"wandb.tags={format_override_value(tags)}")
    return overrides


def _dict_to_overrides(prefix: str, payload: dict[str, Any]) -> list[str]:
    return [f"{prefix}.{key}={format_override_value(value)}" for key, value in payload.items()]


def build_description(config: ResolvedTaskConfig) -> str:
    mechanism_label = config.sampling.mechanism_label or config.sampling.key
    mechanism_summary = config.sampling.mechanism_summary or config.sampling.update_rule
    parameterization_label = config.parameterization.label or config.parameterization.key
    parameterization_summary = config.parameterization.summary or config.parameterization.parameterization
    rank_text = basis_rank_tag(config.parameterization)
    factor_text = factor_rank_tag(config.parameterization)
    if rank_text and factor_text:
        rank_text = f"{rank_text}, {factor_text}"
    elif factor_text:
        rank_text = factor_text
    basis_clause = f", basis={rank_text}" if rank_text else ""
    return (
        f"{config.dataset.description_label} vLLM Spectral-ES run with {parameterization_label} "
        f"({parameterization_summary}){basis_clause}, "
        f"K={config.model.num_mutants}, q_batch={config.model.effective_question_batch}, "
        f"micro_batch={config.train_micro_batch}, chunk={config.mutant_chunk_size}, "
        f"sampling={mechanism_label} ({mechanism_summary}), eval split {config.params.eval_split}, "
        f"thinking={config.model.thinking}, "
        f"gpu_count={config.gpu_count}."
    )


def build_payload(config: ResolvedTaskConfig, *, task_name: str, run_id: str, description: str) -> dict[str, Any]:
    override_flags = " ".join(f'--override "{item}"' for item in build_overrides(config, run_id))
    return {
        "TaskName": task_name,
        "Description": description,
        "Entrypoint": (
            "source /GenSIvePFS/users/yfwang/miniconda3/etc/profile.d/conda.sh\n"
            "conda activate verl\n\n"
            f"cd {PROJECT_DIR}\n"
            "export PYTHONUNBUFFERED=1\n"
            "export TOKENIZERS_PARALLELISM=false\n"
            f"torchrun --standalone --nproc_per_node={config.world_size} train.py "
            f"--config {config.dataset.base_config_path} {override_flags}\n"
        ),
        "ImageUrl": "vemlp-cn-beijing.cr.volces.com/preset-images/cuda:12.8.1-py3.12-ubuntu22.04",
        "ResourceQueueID": "q-20250929101814-vmcz2",
        "ResourceQueueName": "sia-thu",
        "Framework": "Custom",
        "Envs": [
            {
                "Name": "WANDB_API_KEY",
                "Value": "d4bfc5576f50cdd6dc70acca17134f341fee3906",
                "IsPrivate": True,
            }
        ],
        "TaskRoleSpecs": [
            {
                "RoleName": "worker",
                "RoleReplicas": 1,
                "Flavor": config.flavor,
            }
        ],
        "ActiveDeadlineSeconds": 864000,
        "DelayExitTimeSeconds": 0,
        "AccessType": "Public",
        "Preemptible": False,
        "Priority": 4,
        "RetryOptions": {
            "EnableRetry": False,
            "EnableReserveResourceOnRetry": False,
        },
        "DiagOptions": [
            {"Name": "EnvironmentalDiagnosis", "Enable": False},
            {"Name": "PythonDetection", "Enable": False},
            {"Name": "LogDetection", "Enable": False},
        ],
        "EnableTensorBoard": False,
        "Storages": [
            {
                "Type": "Vepfs",
                "MountPath": "/GenSIvePFS/users/model",
                "SubPath": "users/model",
            },
            {
                "Type": "Vepfs",
                "MountPath": "/GenSIvePFS/users/yfwang",
                "SubPath": "users/yfwang",
            },
        ],
    }


def build_task_artifacts(selection: TaskSelection, *, timestamp: str | None = None) -> tuple[ResolvedTaskConfig, str, str, Path, dict[str, Any]]:
    resolved = resolve_task_selection(selection)
    ts = timestamp or timestamp_now()
    run_id = build_run_suffix(resolved, ts)
    task_name = f"lowrank-spectral-es-{run_id}"
    config_dir = VOLC_CONFIG_ROOT / resolved.dataset.task_config_subdir
    config_path = config_dir / f"{run_id}.yaml"
    payload = build_payload(resolved, task_name=task_name, run_id=run_id, description=build_description(resolved))
    return resolved, task_name, run_id, config_path, payload


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def submit_config(conf_path: Path) -> str:
    proc = subprocess.run(
        ["volc", "ml_task", "submit", "--conf", str(conf_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.stdout.strip():
        print(proc.stdout.strip())
    if proc.stderr.strip():
        print(proc.stderr.strip())
    if proc.returncode != 0:
        raise RuntimeError(f"submit failed for {conf_path}")
    raw = "\n".join([proc.stdout, proc.stderr])
    match = re.search(r"task_id=(t-[A-Za-z0-9-]+)", raw)
    if not match:
        raise RuntimeError("submit succeeded but task_id was not found in CLI output")
    return match.group(1)


def preview_submission(selection: TaskSelection, *, timestamp: str | None = None) -> tuple[ResolvedTaskConfig, str, str, Path, dict[str, Any]]:
    resolved, task_name, run_id, config_path, payload = build_task_artifacts(selection, timestamp=timestamp)
    preview = {
        "dataset": resolved.dataset.key,
        "model": resolved.model.key,
        "parameterization": resolved.parameterization.key,
        "params": resolved.params.key,
        "sampling": resolved.sampling.key,
        "gpu_count": resolved.gpu_count,
        "world_size": resolved.world_size,
        "flavor": resolved.flavor,
        "mutants_per_worker": resolved.mutants_per_worker,
        "mutant_chunk_size": resolved.mutant_chunk_size,
        "train_micro_batch": resolved.train_micro_batch,
        "task_name": task_name,
        "run_id": run_id,
        "config_path": str(config_path),
        "base_config_path": resolved.dataset.base_config_path,
        "model_path": resolved.model.model_path,
        "num_mutants": resolved.model.num_mutants,
        "effective_question_batch": resolved.model.effective_question_batch,
        "subspace_parameterization": resolved.parameterization.parameterization,
        "subspace_factor_rank": resolved.parameterization.factor_rank,
        "sigma_m": resolved.parameterization.sigma_m,
        "cma_selection_ratio": resolved.params.cma_selection_ratio,
        "cma_mean_step_scale": resolved.params.cma_mean_step_scale,
        "update_rule": resolved.sampling.update_rule,
        "antithetic": resolved.sampling.antithetic,
        "thinking": resolved.model.thinking,
        "target_blocks": resolved.params.target_blocks,
    }
    if resolved.sampling.alpha_m is not None:
        preview["alpha_m"] = resolved.sampling.alpha_m
    if resolved.parameterization.subspace_rank is not None:
        preview["subspace_rank"] = resolved.parameterization.subspace_rank
    print(yaml.safe_dump(preview, sort_keys=False), end="")
    return resolved, task_name, run_id, config_path, payload
