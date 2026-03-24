from __future__ import annotations

from typing import Any

import torch

from es.updater import resolve_named_value


def _clone_payload_like(
    payloads: dict[str, dict[str, torch.Tensor]],
    *,
    fill_value: float = 0.0,
) -> dict[str, dict[str, torch.Tensor]]:
    cloned: dict[str, dict[str, torch.Tensor]] = {}
    for name, bundle in payloads.items():
        cloned[name] = {}
        for key, tensor in bundle.items():
            cloned[name][key] = torch.full_like(tensor[0], fill_value=fill_value, dtype=torch.float32)
    return cloned


def payload_global_norm(payloads: dict[str, dict[str, torch.Tensor]]) -> float:
    total = 0.0
    for bundle in payloads.values():
        for tensor in bundle.values():
            total += float(torch.sum(tensor.float() ** 2).item())
    return total ** 0.5


def payload_max_norm(payloads: dict[str, dict[str, torch.Tensor]]) -> float:
    best = 0.0
    for bundle in payloads.values():
        for tensor in bundle.values():
            best = max(best, float(torch.linalg.vector_norm(tensor.float()).item()))
    return best


def compute_pairwise_direction_payloads(
    *,
    noise_payloads: dict[str, dict[str, torch.Tensor]],
    rewards: torch.Tensor,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, float]]:
    rewards = rewards.float()
    if rewards.numel() % 2 != 0:
        raise ValueError("pairwise directional update requires an even number of rewards")
    half = rewards.numel() // 2
    if half == 0:
        directions = _clone_payload_like(noise_payloads)
        return directions, {
            "pair_diff_mean": 0.0,
            "pair_diff_std": 0.0,
            "pair_diff_abs_mean": 0.0,
            "pair_nonzero_rate": 0.0,
            "direction_global_norm": 0.0,
        }

    pair_diffs = rewards[:half] - rewards[half:]
    pair_advantages = 0.5 * pair_diffs
    directions = _clone_payload_like(noise_payloads)
    for name, bundle in noise_payloads.items():
        for key, tensor in bundle.items():
            plus_noise = tensor[:half].float()
            directions[name][key] = torch.tensordot(pair_advantages, plus_noise, dims=([0], [0])) / float(half)

    stats = {
        "pair_diff_mean": float(pair_diffs.mean().item()),
        "pair_diff_std": float(pair_diffs.std(unbiased=False).item()),
        "pair_diff_abs_mean": float(pair_diffs.abs().mean().item()),
        "pair_advantage_mean": float(pair_advantages.mean().item()),
        "pair_advantage_std": float(pair_advantages.std(unbiased=False).item()),
        "pair_nonzero_rate": float((pair_diffs != 0).float().mean().item()),
        "direction_global_norm": payload_global_norm(directions),
    }
    return directions, stats


def compute_gaussian_direction_payloads(
    *,
    noise_payloads: dict[str, dict[str, torch.Tensor]],
    rewards: torch.Tensor,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, float]]:
    rewards = rewards.float()
    count = rewards.numel()
    if count == 0:
        directions = _clone_payload_like(noise_payloads)
        return directions, {
            "reward_mean": 0.0,
            "reward_std": 0.0,
            "reward_abs_mean": 0.0,
            "reward_nonzero_rate": 0.0,
            "direction_global_norm": 0.0,
        }

    directions = _clone_payload_like(noise_payloads)
    for name, bundle in noise_payloads.items():
        for key, tensor in bundle.items():
            directions[name][key] = torch.tensordot(rewards, tensor.float(), dims=([0], [0])) / float(count)

    stats = {
        "reward_mean": float(rewards.mean().item()),
        "reward_std": float(rewards.std(unbiased=False).item()),
        "reward_abs_mean": float(rewards.abs().mean().item()),
        "reward_nonzero_rate": float((rewards != 0).float().mean().item()),
        "direction_global_norm": payload_global_norm(directions),
    }
    return directions, stats


def apply_alpha_update_to_direction_payloads(
    *,
    direction_payloads: dict[str, dict[str, torch.Tensor]],
    alpha_config: Any,
    sigma_config: Any,
    max_layer_step_config: Any | None = None,
    eps: float = 1e-8,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, float]]:
    alpha = resolve_named_value(alpha_config, "m")
    sigma = resolve_named_value(sigma_config, "m")
    if alpha <= 0 or sigma <= 0:
        return _clone_payload_like(direction_payloads), {
            "alpha": max(alpha, 0.0),
            "sigma": max(sigma, 0.0),
            "alpha_over_sigma": 0.0,
            "raw_step_global_norm": 0.0,
            "step_global_norm": 0.0,
            "step_layer_max_norm": 0.0,
            "step_layers_clipped": 0.0,
        }
    alpha_over_sigma = alpha / sigma
    step_payloads = _clone_payload_like(direction_payloads)
    raw_step_payloads = _clone_payload_like(direction_payloads)
    layers_clipped = 0
    max_layer_step_norm = None
    if max_layer_step_config is not None:
        max_layer_step_norm = resolve_named_value(max_layer_step_config, "m")
        if max_layer_step_norm <= 0:
            max_layer_step_norm = None

    for name, bundle in direction_payloads.items():
        for key, tensor in bundle.items():
            step_tensor = tensor.float() * alpha_over_sigma
            raw_step_payloads[name][key] = step_tensor.clone()
            if max_layer_step_norm is not None:
                layer_norm = float(torch.linalg.vector_norm(step_tensor).item())
                if layer_norm > max_layer_step_norm and layer_norm > eps:
                    step_tensor = step_tensor * (max_layer_step_norm / layer_norm)
                    layers_clipped += 1
            step_payloads[name][key] = step_tensor

    stats = {
        "alpha": alpha,
        "sigma": sigma,
        "alpha_over_sigma": alpha_over_sigma,
        "raw_step_global_norm": payload_global_norm(raw_step_payloads),
        "step_global_norm": payload_global_norm(step_payloads),
        "step_layer_max_norm": payload_max_norm(step_payloads),
        "step_layers_clipped": float(layers_clipped),
    }
    return step_payloads, stats


def scale_direction_payloads(
    *,
    direction_payloads: dict[str, dict[str, torch.Tensor]],
    step_config: Any,
    step_scale: float = 1.0,
    max_layer_step_config: Any | None = None,
    eps: float = 1e-8,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, float]]:
    scaled_config = resolve_named_value(step_config, "m") * float(step_scale)
    return apply_alpha_update_to_direction_payloads(
        direction_payloads=direction_payloads,
        alpha_config={"m": scaled_config},
        sigma_config={"m": 1.0},
        max_layer_step_config=max_layer_step_config,
        eps=eps,
    )
