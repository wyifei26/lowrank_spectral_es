from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from models.layer_selector import LayerSelection
from models.spectral_es import (
    DiagonalSpectralAdapterLayer,
    FactorizedSpectralAdapterLayer,
    LoRAESAdapterLayer,
    SpectralAdapterLayer,
)


PARAMETERIZATION_SPECTRAL_DENSE = "spectral_dense"
PARAMETERIZATION_SPECTRAL_DIAGONAL = "spectral_diagonal"
PARAMETERIZATION_LORA_ES = "lora_es"
PARAMETERIZATION_FULL_FACTORIZED_M = "full_factorized_m"
SUPPORTED_PARAMETERIZATIONS = {
    PARAMETERIZATION_SPECTRAL_DENSE,
    PARAMETERIZATION_SPECTRAL_DIAGONAL,
    PARAMETERIZATION_LORA_ES,
    PARAMETERIZATION_FULL_FACTORIZED_M,
}


class SpectralVLLMState:
    def __init__(
        self,
        *,
        algorithm_name: str,
        selections: list[LayerSelection],
        cache_payload: dict[str, dict[str, torch.Tensor]],
        parameterization: str,
        factor_rank: int | None = None,
        factor_init_scale: float = 0.01,
        init_method: str = "none",
        init_rho: float = 0.0,
    ) -> None:
        if algorithm_name != "spectral_es":
            raise ValueError(f"vLLM backend currently only supports spectral_es, got {algorithm_name}")
        if parameterization not in SUPPORTED_PARAMETERIZATIONS:
            raise ValueError(
                f"unsupported subspace.parameterization: {parameterization}; "
                f"expected one of {sorted(SUPPORTED_PARAMETERIZATIONS)}"
            )
        self.algorithm_name = algorithm_name
        self.cache_payload = cache_payload
        self.parameterization = parameterization
        self.factor_rank = factor_rank
        self.factor_init_scale = float(factor_init_scale)
        self.adapters: dict[
            str,
            SpectralAdapterLayer | DiagonalSpectralAdapterLayer | FactorizedSpectralAdapterLayer | LoRAESAdapterLayer,
        ] = {}
        self.target_modules = sorted({selection.module_key for selection in selections})

        for selection in selections:
            cache_entry = cache_payload[selection.full_name]
            if parameterization == PARAMETERIZATION_SPECTRAL_DENSE:
                adapter = SpectralAdapterLayer(
                    layer_name=selection.full_name,
                    u=cache_entry["u"].float(),
                    vh=cache_entry["vh"].float(),
                    singular_values=cache_entry["s"].float(),
                    init_method=init_method,
                    init_rho=init_rho,
                )
            elif parameterization == PARAMETERIZATION_SPECTRAL_DIAGONAL:
                adapter = DiagonalSpectralAdapterLayer(
                    layer_name=selection.full_name,
                    u=cache_entry["u"].float(),
                    vh=cache_entry["vh"].float(),
                    singular_values=cache_entry["s"].float(),
                    init_method=init_method,
                    init_rho=init_rho,
                )
            elif parameterization == PARAMETERIZATION_FULL_FACTORIZED_M:
                if factor_rank is None or int(factor_rank) <= 0:
                    raise ValueError("full_factorized_m requires subspace.factor_rank > 0")
                adapter = FactorizedSpectralAdapterLayer(
                    layer_name=selection.full_name,
                    u=cache_entry["u"].float(),
                    vh=cache_entry["vh"].float(),
                    factor_rank=int(factor_rank),
                    init_scale=self.factor_init_scale,
                )
            else:
                if factor_rank is None or int(factor_rank) <= 0:
                    raise ValueError("lora_es requires subspace.factor_rank > 0")
                adapter = LoRAESAdapterLayer(
                    layer_name=selection.full_name,
                    out_dim=int(selection.module.out_features),
                    in_dim=int(selection.module.in_features),
                    factor_rank=int(factor_rank),
                    init_scale=self.factor_init_scale,
                )
            self.adapters[selection.full_name] = adapter

        self.export_rank = max((adapter.export_lora_rank() for adapter in self.adapters.values()), default=0)
        self.activate_current_state()

    def activate_current_state(self) -> None:
        for adapter in self.adapters.values():
            adapter.activate_current()

    def activate_mutants(self, noise_payloads: dict[str, dict[str, torch.Tensor]], sigma_config: Any) -> None:
        for name, adapter in self.adapters.items():
            adapter.activate_mutants(noise_payloads[name], sigma_config)

    def sample_noise(self, num_mutants: int, *, antithetic: bool = True) -> dict[str, dict[str, torch.Tensor]]:
        return {
            name: adapter.sample_noise(num_mutants, antithetic=antithetic)
            for name, adapter in self.adapters.items()
        }

    def apply_es_update(
        self,
        *,
        utilities: torch.Tensor,
        noise_payloads: dict[str, dict[str, torch.Tensor]],
        alpha_config: Any,
        sigma_config: Any,
    ) -> None:
        for name, adapter in self.adapters.items():
            adapter.apply_es_update(
                utilities=utilities,
                noise_bundle=noise_payloads[name],
                alpha_config=alpha_config,
                sigma_config=sigma_config,
            )

    def adapter_state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        return {name: adapter.export_trainable_state() for name, adapter in self.adapters.items()}

    def load_adapter_state_dict(self, payload: dict[str, dict[str, torch.Tensor]]) -> None:
        for name, state in payload.items():
            if name not in self.adapters:
                continue
            self.adapters[name].load_trainable_state(state)
        self.activate_current_state()

    def adapter_norms(self) -> dict[str, float]:
        return {name: adapter.state_norm() for name, adapter in self.adapters.items()}

    def initial_noise_scales(self) -> dict[str, torch.Tensor]:
        payload: dict[str, torch.Tensor] = {}
        for name, adapter in self.adapters.items():
            scale = adapter.initial_noise_scale()
            if scale is not None:
                payload[name] = scale
        return payload

    def apply_step_payloads(
        self,
        step_payloads: dict[str, dict[str, torch.Tensor]],
        *,
        max_state_norm: float | None = None,
    ) -> None:
        for name, adapter in self.adapters.items():
            bundle = step_payloads.get(name)
            if not bundle:
                continue
            adapter.apply_step_payload(bundle, max_state_norm=max_state_norm)

    def export_adapter(
        self,
        *,
        output_dir: str | Path,
        base_model_name_or_path: str,
        adapter_name: str,
        active_index: int | None = None,
        clean_dir: bool = True,
    ) -> Path:
        output_dir = Path(output_dir)
        if clean_dir and output_dir.exists():
            for child in output_dir.iterdir():
                if child.is_file() or child.is_symlink():
                    child.unlink()
                else:
                    for nested in child.rglob("*"):
                        if nested.is_file() or nested.is_symlink():
                            nested.unlink()
                    for nested in sorted(child.rglob("*"), reverse=True):
                        if nested.is_dir():
                            nested.rmdir()
                    child.rmdir()
        output_dir.mkdir(parents=True, exist_ok=True)

        state_dict: dict[str, torch.Tensor] = {}
        for full_name, adapter in self.adapters.items():
            lora_a, lora_b = adapter.export_lora_weights(active_index=active_index)
            state_dict[f"base_model.model.{full_name}.lora_A.weight"] = lora_a.cpu()
            state_dict[f"base_model.model.{full_name}.lora_B.weight"] = lora_b.cpu()

        save_file(state_dict, str(output_dir / "adapter_model.safetensors"))
        config = {
            "base_model_name_or_path": base_model_name_or_path,
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": True,
            "lora_alpha": self.export_rank,
            "lora_dropout": 0.0,
            "peft_type": "LORA",
            "r": self.export_rank,
            "target_modules": self.target_modules,
            "task_type": "CAUSAL_LM",
            "adapter_name": adapter_name,
        }
        with (output_dir / "adapter_config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, ensure_ascii=True)
        return output_dir


def build_vllm_spectral_state(
    *,
    algorithm_name: str,
    selections: list[LayerSelection],
    cache_payload: dict[str, dict[str, torch.Tensor]],
    subspace_config: dict[str, Any] | None = None,
) -> SpectralVLLMState:
    subspace_config = dict(subspace_config or {})
    parameterization = str(subspace_config.get("parameterization", PARAMETERIZATION_SPECTRAL_DENSE)).strip().lower()
    factor_rank_raw = subspace_config.get("factor_rank")
    factor_rank = None if factor_rank_raw in (None, 0) else int(factor_rank_raw)
    factor_init_scale = float(subspace_config.get("factor_init_scale", 0.01))
    init_method = str(
        subspace_config.get("init_method", subspace_config.get("diagonal_init_method", "none"))
    ).strip().lower()
    init_rho = float(subspace_config.get("init_rho", subspace_config.get("diagonal_init_rho", 0.0)))
    return SpectralVLLMState(
        algorithm_name=algorithm_name,
        selections=selections,
        cache_payload=cache_payload,
        parameterization=parameterization,
        factor_rank=factor_rank,
        factor_init_scale=factor_init_scale,
        init_method=init_method,
        init_rho=init_rho,
    )


def cleanup_cpu_model(model: torch.nn.Module | None) -> None:
    if model is None:
        return
    del model
    gc.collect()
