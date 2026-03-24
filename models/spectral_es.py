from __future__ import annotations

from typing import Any

import torch

from es.noise import sample_antithetic_normal, sample_standard_normal
from es.updater import resolve_named_value
from models.mutant_linear import AdapterLayerBase


class SpectralAdapterLayer(AdapterLayerBase):
    def __init__(self, *, layer_name: str, u: torch.Tensor, vh: torch.Tensor):
        super().__init__(layer_name=layer_name)
        self.register_buffer("u_basis", u.float())
        self.register_buffer("v_basis", vh.float().transpose(0, 1).contiguous())
        rank = vh.shape[0]
        self.register_buffer("m_state", torch.zeros(rank, rank, dtype=torch.float32, device=u.device))
        self.active_m: torch.Tensor | None = None

    def sample_noise(self, num_mutants: int, *, antithetic: bool = True) -> dict[str, torch.Tensor]:
        sampler = sample_antithetic_normal if antithetic else sample_standard_normal
        return {
            "m": sampler(
                shape=tuple(self.m_state.shape),
                num_mutants=num_mutants,
                device=self.m_state.device,
                dtype=torch.float32,
            )
        }

    def activate_mutants(self, noise_bundle: dict[str, torch.Tensor], sigma_config: Any) -> None:
        sigma = resolve_named_value(sigma_config, "m")
        self.active_m = self.m_state.unsqueeze(0) + sigma * noise_bundle["m"]

    def activate_current(self) -> None:
        self.active_m = self.m_state.unsqueeze(0)

    def clear_active(self) -> None:
        self.active_m = None

    def forward_delta(self, hidden_states: torch.Tensor, mutant_indices: torch.Tensor) -> torch.Tensor:
        if self.active_m is None:
            return torch.zeros(
                (*hidden_states.shape[:-1], self.u_basis.shape[0]),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        matrices = self.active_m[mutant_indices].to(hidden_states.dtype)
        v_basis = self.v_basis.to(hidden_states.dtype)
        u_basis = self.u_basis.to(hidden_states.dtype)
        proj = torch.einsum("bsi,ir->bsr", hidden_states, v_basis)
        mixed = torch.einsum("bsr,brq->bsq", proj, matrices.transpose(1, 2))
        return torch.einsum("bsr,or->bso", mixed, u_basis)

    def apply_es_update(
        self,
        *,
        utilities: torch.Tensor,
        noise_bundle: dict[str, torch.Tensor],
        alpha_config: Any,
        sigma_config: Any,
    ) -> None:
        raise RuntimeError("legacy ES update path has been removed; use pairwise directional updates instead")

    def state_norm(self) -> float:
        return float(torch.linalg.vector_norm(self.m_state).item())

    def apply_step(self, step_tensor: torch.Tensor, *, max_state_norm: float | None = None) -> None:
        self.m_state.add_(step_tensor.to(self.m_state.dtype))
        if max_state_norm is None or max_state_norm <= 0:
            return
        state_norm = self.state_norm()
        if state_norm > max_state_norm and state_norm > 0:
            self.m_state.mul_(max_state_norm / state_norm)

    def apply_step_payload(
        self,
        step_bundle: dict[str, torch.Tensor],
        *,
        max_state_norm: float | None = None,
    ) -> None:
        if "m" not in step_bundle:
            return
        self.apply_step(step_bundle["m"], max_state_norm=max_state_norm)
