from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from es.updater import resolve_named_value
from models.layer_selector import LayerSelection
from models.spectral_es import SpectralAdapterLayer


def _factorize_square_matrix(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"expected square matrix, got shape={tuple(matrix.shape)}")
    u_m, s_m, vh_m = torch.linalg.svd(matrix.float(), full_matrices=False)
    sqrt_s = torch.sqrt(torch.clamp_min(s_m, 0.0))
    left = u_m * sqrt_s.unsqueeze(0)
    right = sqrt_s.unsqueeze(1) * vh_m
    return left.contiguous(), right.contiguous()


def spectral_matrix_to_lora(
    *,
    u_basis: torch.Tensor,
    vh_basis: torch.Tensor,
    matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    left, right = _factorize_square_matrix(matrix)
    lora_b = (u_basis.float() @ left).contiguous()
    lora_a = (right @ vh_basis.float()).contiguous()
    return lora_a, lora_b


class SpectralVLLMState:
    def __init__(
        self,
        *,
        algorithm_name: str,
        selections: list[LayerSelection],
        cache_payload: dict[str, dict[str, torch.Tensor]],
    ) -> None:
        if algorithm_name != "spectral_es":
            raise ValueError(f"vLLM backend currently only supports spectral_es, got {algorithm_name}")
        self.algorithm_name = algorithm_name
        self.cache_payload = cache_payload
        self.adapters: dict[str, SpectralAdapterLayer] = {}
        self.target_modules = sorted({selection.module_key for selection in selections})
        for selection in selections:
            cache_entry = cache_payload[selection.full_name]
            self.adapters[selection.full_name] = SpectralAdapterLayer(
                layer_name=selection.full_name,
                u=cache_entry["u"].float(),
                vh=cache_entry["vh"].float(),
            )
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
        return {
            name: {"m_state": adapter.m_state.detach().cpu().clone()}
            for name, adapter in self.adapters.items()
        }

    def load_adapter_state_dict(self, payload: dict[str, dict[str, torch.Tensor]]) -> None:
        for name, state in payload.items():
            self.adapters[name].m_state.copy_(state["m_state"].to(self.adapters[name].m_state.device))
        self.activate_current_state()

    def adapter_norms(self) -> dict[str, float]:
        return {name: adapter.state_norm() for name, adapter in self.adapters.items()}

    def apply_step_payloads(
        self,
        step_payloads: dict[str, dict[str, torch.Tensor]],
        *,
        max_state_norm: float | None = None,
    ) -> None:
        for name, adapter in self.adapters.items():
            bundle = step_payloads.get(name)
            if not bundle or "m" not in bundle:
                continue
            adapter.apply_step(bundle["m"], max_state_norm=max_state_norm)

    def export_adapter(
        self,
        *,
        output_dir: str | Path,
        base_model_name_or_path: str,
        rank: int,
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
            matrix = adapter.m_state if active_index is None else adapter.active_m[active_index]
            lora_a, lora_b = spectral_matrix_to_lora(
                u_basis=self.cache_payload[full_name]["u"],
                vh_basis=self.cache_payload[full_name]["vh"],
                matrix=matrix,
            )
            state_dict[f"base_model.model.{full_name}.lora_A.weight"] = lora_a.cpu()
            state_dict[f"base_model.model.{full_name}.lora_B.weight"] = lora_b.cpu()

        save_file(state_dict, str(output_dir / "adapter_model.safetensors"))
        config = {
            "base_model_name_or_path": base_model_name_or_path,
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": True,
            "lora_alpha": rank,
            "lora_dropout": 0.0,
            "peft_type": "LORA",
            "r": rank,
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
) -> SpectralVLLMState:
    return SpectralVLLMState(
        algorithm_name=algorithm_name,
        selections=selections,
        cache_payload=cache_payload,
    )


def cleanup_cpu_model(model: torch.nn.Module | None) -> None:
    if model is None:
        return
    del model
    gc.collect()
