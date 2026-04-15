from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from es.spectral_update import payload_global_norm, payload_max_norm
from es.updater import resolve_named_value


def _resolve_positive_float(config: dict[str, Any], key: str, default: float) -> float:
    value = config.get(key, default)
    value = float(value)
    return value if value > 0.0 else default


def _resolve_optional_positive_float(config: dict[str, Any], key: str) -> float | None:
    if key not in config or config[key] in (None, 0):
        return None
    value = float(config[key])
    return value if value > 0.0 else None


def _standard_recombination_weights(population_size: int, *, selection_ratio: float) -> torch.Tensor:
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    clamped_ratio = min(max(selection_ratio, 0.0), 1.0)
    mu = max(1, min(population_size, int(math.floor(population_size * clamped_ratio))))
    ranks = torch.arange(1, mu + 1, dtype=torch.float32)
    weights = torch.log(torch.tensor(mu + 0.5, dtype=torch.float32)) - torch.log(ranks)
    weights = torch.clamp_min(weights, 0.0)
    weights_sum = float(weights.sum().item())
    if weights_sum <= 0.0:
        weights = torch.full((mu,), 1.0 / float(mu), dtype=torch.float32)
    else:
        weights /= weights_sum
    return weights


def _layer_chi_norm(dim: int) -> float:
    return math.sqrt(float(dim)) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))


@dataclass
class LayerCMAState:
    sigma: torch.Tensor
    cov_diag: torch.Tensor
    p_sigma: torch.Tensor
    p_c: torch.Tensor
    generation: int = 0


class PerLayerCMAES:
    def __init__(
        self,
        *,
        layer_shapes: dict[str, torch.Size],
        sigma_config: Any,
        cma_config: dict[str, Any] | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ) -> None:
        self.layer_shapes = {name: torch.Size(shape) for name, shape in layer_shapes.items()}
        self.dtype = dtype
        self.device = device or torch.device("cpu")
        self.cma_config = dict(cma_config or {})
        self.initial_sigma = resolve_named_value(sigma_config, "m")
        if self.initial_sigma <= 0.0:
            raise ValueError("per-layer CMA-ES requires es.sigma.m > 0")
        self.selection_ratio = _resolve_positive_float(self.cma_config, "selection_ratio", 0.5)
        self.mean_step_scale = _resolve_positive_float(self.cma_config, "mean_step_scale", 1.0)
        self.min_sigma = _resolve_positive_float(self.cma_config, "min_sigma", 1e-6)
        self.max_sigma = _resolve_optional_positive_float(self.cma_config, "max_sigma")
        self.min_eigenvalue = _resolve_positive_float(self.cma_config, "min_eigenvalue", 1e-8)
        self.layers: dict[str, LayerCMAState] = {}
        for name, shape in self.layer_shapes.items():
            dim = math.prod(shape)
            self.layers[name] = LayerCMAState(
                sigma=torch.tensor(float(self.initial_sigma), dtype=self.dtype, device=self.device),
                cov_diag=torch.ones(dim, dtype=self.dtype, device=self.device),
                p_sigma=torch.zeros(dim, dtype=self.dtype, device=self.device),
                p_c=torch.zeros(dim, dtype=self.dtype, device=self.device),
            )

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "selection_ratio": self.selection_ratio,
                "mean_step_scale": self.mean_step_scale,
                "min_sigma": self.min_sigma,
                "max_sigma": self.max_sigma,
                "min_eigenvalue": self.min_eigenvalue,
                "initial_sigma": self.initial_sigma,
            },
            "layers": {
                name: {
                    "sigma": layer.sigma.detach().cpu().clone(),
                    "cov_diag": layer.cov_diag.detach().cpu().clone(),
                    "p_sigma": layer.p_sigma.detach().cpu().clone(),
                    "p_c": layer.p_c.detach().cpu().clone(),
                    "generation": int(layer.generation),
                }
                for name, layer in self.layers.items()
            },
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        layer_payloads = payload.get("layers", {})
        for name, state in layer_payloads.items():
            if name not in self.layers:
                continue
            layer = self.layers[name]
            layer.sigma = state["sigma"].to(device=self.device, dtype=self.dtype)
            cov_diag = state.get("cov_diag")
            if cov_diag is None:
                raise ValueError("diagonal CMA checkpoint is missing cov_diag")
            layer.cov_diag = cov_diag.to(device=self.device, dtype=self.dtype)
            layer.p_sigma = state["p_sigma"].to(device=self.device, dtype=self.dtype)
            layer.p_c = state["p_c"].to(device=self.device, dtype=self.dtype)
            layer.generation = int(state.get("generation", 0))

    def sample_noise(self, num_mutants: int, *, antithetic: bool = True) -> dict[str, dict[str, torch.Tensor]]:
        if antithetic and num_mutants % 2 != 0:
            raise ValueError("per-layer diagonal CMA-ES with antithetic sampling requires an even num_mutants")
        if num_mutants <= 0:
            raise ValueError("num_mutants must be positive")

        noise_payloads: dict[str, dict[str, torch.Tensor]] = {}
        for name, shape in self.layer_shapes.items():
            layer = self.layers[name]
            dim = math.prod(shape)
            if antithetic:
                half = num_mutants // 2
                z_base = torch.randn((half, dim), dtype=self.dtype, device=self.device)
                z = torch.cat([z_base, -z_base], dim=0)
            else:
                z = torch.randn((num_mutants, dim), dtype=self.dtype, device=self.device)
            transformed = z * torch.sqrt(torch.clamp_min(layer.cov_diag, self.min_eigenvalue)).unsqueeze(0)
            noise_payloads[name] = {
                "m": transformed.reshape(num_mutants, *shape).contiguous(),
                "m_z": z.reshape(num_mutants, *shape).contiguous(),
                "sigma": layer.sigma.detach().clone(),
            }
        return noise_payloads

    def apply_update(
        self,
        *,
        rewards: torch.Tensor,
        noise_payloads: dict[str, dict[str, torch.Tensor]],
        current_states: dict[str, torch.Tensor],
        max_layer_step_config: Any | None = None,
        max_state_norm: float | None = None,
        eps: float = 1e-8,
    ) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, float], dict[str, float]]:
        rewards = rewards.float()
        population_size = rewards.numel()
        if population_size <= 0:
            raise ValueError("per-layer diagonal CMA-ES requires at least one reward")
        sorted_indices = torch.argsort(rewards, descending=True)
        weights = _standard_recombination_weights(population_size, selection_ratio=self.selection_ratio)
        mu = weights.numel()
        top_indices = sorted_indices[:mu]
        weight_device = weights.to(device=self.device, dtype=self.dtype)
        mu_eff = float(1.0 / torch.sum(weight_device**2).item())

        step_payloads: dict[str, dict[str, torch.Tensor]] = {}
        raw_step_payloads: dict[str, dict[str, torch.Tensor]] = {}
        sigma_values: list[float] = []
        cov_traces: list[float] = []
        step_norms: list[float] = []
        raw_step_norms: list[float] = []
        hsigma_flags: list[float] = []
        layers_clipped = 0.0
        selected_reward_mean = float(rewards[top_indices].mean().item())
        selected_reward_std = float(rewards[top_indices].std(unbiased=False).item()) if mu > 1 else 0.0
        reward_best = float(rewards[top_indices[0]].item())

        max_layer_step_norm = None
        if max_layer_step_config is not None:
            max_layer_step_norm = resolve_named_value(max_layer_step_config, "m")
            if max_layer_step_norm <= 0.0:
                max_layer_step_norm = None

        for name, shape in self.layer_shapes.items():
            layer = self.layers[name]
            dim = math.prod(shape)
            y_all = noise_payloads[name]["m"].reshape(population_size, dim).to(device=self.device, dtype=self.dtype)
            z_all = noise_payloads[name]["m_z"].reshape(population_size, dim).to(device=self.device, dtype=self.dtype)
            y_top = y_all[top_indices.to(device=self.device)]
            z_top = z_all[top_indices.to(device=self.device)]
            y_w = torch.sum(y_top * weight_device.unsqueeze(1), dim=0)
            z_w = torch.sum(z_top * weight_device.unsqueeze(1), dim=0)
            raw_step_flat = layer.sigma * self.mean_step_scale * y_w
            raw_step_norm = float(torch.linalg.vector_norm(raw_step_flat).item())
            raw_step_norms.append(raw_step_norm)

            step_flat = raw_step_flat.clone()
            if max_layer_step_norm is not None and raw_step_norm > max_layer_step_norm and raw_step_norm > eps:
                step_flat.mul_(max_layer_step_norm / raw_step_norm)
                layers_clipped += 1.0

            current_flat = current_states[name].reshape(dim).to(device=self.device, dtype=self.dtype)
            next_flat = current_flat + step_flat
            if max_state_norm is not None and max_state_norm > 0.0:
                next_norm = float(torch.linalg.vector_norm(next_flat).item())
                if next_norm > max_state_norm and next_norm > eps:
                    next_flat.mul_(max_state_norm / next_norm)
                    step_flat = next_flat - current_flat

            step_norm = float(torch.linalg.vector_norm(step_flat).item())
            step_norms.append(step_norm)
            step_payloads[name] = {"m": step_flat.reshape(shape).cpu()}
            raw_step_payloads[name] = {"m": raw_step_flat.reshape(shape).cpu()}

            c_sigma, d_sigma, c_c, c1, c_mu = self._learning_rates(dim=dim, mu_eff=mu_eff)
            chi_n = _layer_chi_norm(dim)
            scaled_z_w = self.mean_step_scale * z_w
            scaled_y_w = self.mean_step_scale * y_w
            p_sigma = (1.0 - c_sigma) * layer.p_sigma + math.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * scaled_z_w
            generation = layer.generation + 1
            sigma_norm = float(torch.linalg.vector_norm(p_sigma).item())
            norm_denom = math.sqrt(max(1e-12, 1.0 - (1.0 - c_sigma) ** (2 * generation)))
            h_sigma_threshold = (1.4 + 2.0 / (dim + 1.0)) * chi_n
            h_sigma = 1.0 if sigma_norm / norm_denom < h_sigma_threshold else 0.0
            hsigma_flags.append(h_sigma)
            p_c = (1.0 - c_c) * layer.p_c + h_sigma * math.sqrt(c_c * (2.0 - c_c) * mu_eff) * scaled_y_w
            retained = 1.0 - c1 - c_mu + (1.0 - h_sigma) * c1 * c_c * (2.0 - c_c)
            rank_mu_update_diag = torch.sum((y_top**2) * weight_device.unsqueeze(1), dim=0)
            updated_cov_diag = retained * layer.cov_diag + c1 * (p_c**2) + c_mu * rank_mu_update_diag
            updated_cov_diag = torch.clamp_min(updated_cov_diag, self.min_eigenvalue)
            cov_trace = float(updated_cov_diag.sum().item())
            sigma_scale = math.exp((c_sigma / d_sigma) * (sigma_norm / chi_n - 1.0))
            new_sigma = float(layer.sigma.item()) * sigma_scale
            new_sigma = max(new_sigma, self.min_sigma)
            if self.max_sigma is not None:
                new_sigma = min(new_sigma, self.max_sigma)

            layer.p_sigma = p_sigma
            layer.p_c = p_c
            layer.cov_diag = updated_cov_diag
            layer.sigma = torch.tensor(new_sigma, dtype=self.dtype, device=self.device)
            layer.generation = generation
            sigma_values.append(new_sigma)
            cov_traces.append(cov_trace)

        direction_stats = {
            "cma_selected_reward_mean": selected_reward_mean,
            "cma_selected_reward_std": selected_reward_std,
            "cma_reward_best": reward_best,
            "cma_sigma_mean": sum(sigma_values) / max(len(sigma_values), 1),
            "cma_sigma_min": min(sigma_values) if sigma_values else 0.0,
            "cma_sigma_max": max(sigma_values) if sigma_values else 0.0,
            "cma_cov_trace_mean": sum(cov_traces) / max(len(cov_traces), 1),
            "cma_hsigma_rate": sum(hsigma_flags) / max(len(hsigma_flags), 1),
            "direction_global_norm": payload_global_norm(raw_step_payloads),
        }
        step_stats = {
            "raw_step_global_norm": payload_global_norm(raw_step_payloads),
            "step_global_norm": payload_global_norm(step_payloads),
            "step_layer_max_norm": payload_max_norm(step_payloads),
            "step_layers_clipped": layers_clipped,
            "cma_raw_step_norm_mean": sum(raw_step_norms) / max(len(raw_step_norms), 1),
            "cma_step_norm_mean": sum(step_norms) / max(len(step_norms), 1),
        }
        return step_payloads, direction_stats, step_stats

    @staticmethod
    def _learning_rates(*, dim: int, mu_eff: float) -> tuple[float, float, float, float, float]:
        c_sigma = (mu_eff + 2.0) / (dim + mu_eff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (dim + 1.0)) - 1.0) + c_sigma
        c_c = (4.0 + mu_eff / dim) / (dim + 4.0 + 2.0 * mu_eff / dim)
        c1 = 2.0 / (((dim + 1.3) ** 2) + mu_eff)
        c_mu = min(1.0 - c1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / (((dim + 2.0) ** 2) + mu_eff))
        return c_sigma, d_sigma, c_c, c1, c_mu
