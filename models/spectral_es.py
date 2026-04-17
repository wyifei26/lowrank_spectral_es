from __future__ import annotations

from typing import Any

import torch

from es.noise import sample_antithetic_normal, sample_standard_normal
from es.updater import resolve_named_value
from models.mutant_linear import AdapterLayerBase


def _sample_noise_like(state: torch.Tensor, *, num_mutants: int, antithetic: bool) -> torch.Tensor:
    sampler = sample_antithetic_normal if antithetic else sample_standard_normal
    return sampler(
        shape=tuple(state.shape),
        num_mutants=num_mutants,
        device=state.device,
        dtype=torch.float32,
    )


def _resolve_sigma(noise_bundle: dict[str, Any], sigma_config: Any) -> float:
    sigma_override = noise_bundle.get("sigma")
    if sigma_override is None:
        return resolve_named_value(sigma_config, "m")
    if isinstance(sigma_override, torch.Tensor):
        return float(sigma_override.item())
    return float(sigma_override)


def _resolve_diagonal_init_scale(
    singular_values: torch.Tensor,
    *,
    init_method: str,
    rho: float,
) -> torch.Tensor:
    if init_method == "none":
        return torch.ones_like(singular_values, dtype=torch.float32)
    if init_method != "proportional":
        raise ValueError(f"unsupported diagonal init method: {init_method}")
    if rho < 0.0:
        raise ValueError(f"diagonal proportional rho must be >= 0, got {rho}")
    return (singular_values.float().abs() * float(rho)).contiguous()


class SpectralAdapterLayer(AdapterLayerBase):
    def __init__(self, *, layer_name: str, u: torch.Tensor, vh: torch.Tensor):
        super().__init__(layer_name=layer_name)
        self.register_buffer("u_basis", u.float().contiguous())
        self.register_buffer("vh_basis", vh.float().contiguous())
        self.register_buffer("v_basis", vh.float().transpose(0, 1).contiguous())
        rank = vh.shape[0]
        self.register_buffer("m_state", torch.zeros(rank, rank, dtype=torch.float32, device=u.device))
        self.active_m: torch.Tensor | None = None

    def sample_noise(self, num_mutants: int, *, antithetic: bool = True) -> dict[str, torch.Tensor]:
        return {"m": _sample_noise_like(self.m_state, num_mutants=num_mutants, antithetic=antithetic)}

    def activate_mutants(self, noise_bundle: dict[str, Any], sigma_config: Any) -> None:
        sigma = _resolve_sigma(noise_bundle, sigma_config)
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
        proj = torch.einsum("bsi,ir->bsr", hidden_states, self.v_basis.to(hidden_states.dtype))
        mixed = torch.einsum("bsr,brq->bsq", proj, matrices.transpose(1, 2))
        return torch.einsum("bsr,or->bso", mixed, self.u_basis.to(hidden_states.dtype))

    def apply_es_update(
        self,
        *,
        utilities: torch.Tensor,
        noise_bundle: dict[str, torch.Tensor],
        alpha_config: Any,
        sigma_config: Any,
    ) -> None:
        raise RuntimeError("legacy ES update path has been removed; use payload-based updates instead")

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

    def export_trainable_state(self) -> dict[str, torch.Tensor]:
        return {
            "m_state": self.m_state.detach().cpu().clone(),
            "effective_m_state": self.m_state.detach().cpu().clone(),
        }

    def load_trainable_state(self, payload: dict[str, torch.Tensor]) -> None:
        state = payload.get("m_state")
        if state is None:
            raise KeyError(f"{self.layer_name} checkpoint payload is missing m_state")
        self.m_state.copy_(state.to(self.m_state.device, dtype=self.m_state.dtype))

    def export_lora_weights(self, *, active_index: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        matrix = self.m_state if active_index is None else self.active_m[active_index]
        u_m, s_m, vh_m = torch.linalg.svd(matrix.float(), full_matrices=False)
        sqrt_s = torch.sqrt(torch.clamp_min(s_m, 0.0))
        left = u_m * sqrt_s.unsqueeze(0)
        right = sqrt_s.unsqueeze(1) * vh_m
        lora_b = (self.u_basis.float() @ left).contiguous()
        lora_a = (right @ self.vh_basis.float()).contiguous()
        return lora_a, lora_b

    def export_lora_rank(self) -> int:
        return int(self.m_state.shape[0])

    def effective_matrix(self) -> torch.Tensor:
        return self.m_state.detach().cpu().clone()


class DiagonalSpectralAdapterLayer(AdapterLayerBase):
    def __init__(
        self,
        *,
        layer_name: str,
        u: torch.Tensor,
        vh: torch.Tensor,
        singular_values: torch.Tensor,
        init_method: str = "none",
        init_rho: float = 0.0,
    ):
        super().__init__(layer_name=layer_name)
        self.register_buffer("u_basis", u.float().contiguous())
        self.register_buffer("vh_basis", vh.float().contiguous())
        self.register_buffer("v_basis", vh.float().transpose(0, 1).contiguous())
        self.register_buffer("singular_values", singular_values.float().contiguous())
        rank = int(vh.shape[0])
        self.register_buffer("m_state", torch.zeros(rank, dtype=torch.float32, device=u.device))
        self.register_buffer(
            "noise_scale",
            _resolve_diagonal_init_scale(
                self.singular_values,
                init_method=str(init_method).strip().lower(),
                rho=float(init_rho),
            ),
        )
        self.active_m: torch.Tensor | None = None

    def sample_noise(self, num_mutants: int, *, antithetic: bool = True) -> dict[str, torch.Tensor]:
        base_noise = _sample_noise_like(self.m_state, num_mutants=num_mutants, antithetic=antithetic)
        return {"m": base_noise * self.noise_scale.unsqueeze(0)}

    def activate_mutants(self, noise_bundle: dict[str, Any], sigma_config: Any) -> None:
        sigma = _resolve_sigma(noise_bundle, sigma_config)
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
        diag_entries = self.active_m[mutant_indices].to(hidden_states.dtype)
        proj = torch.einsum("bsi,ir->bsr", hidden_states, self.v_basis.to(hidden_states.dtype))
        scaled = proj * diag_entries
        return torch.einsum("bsr,or->bso", scaled, self.u_basis.to(hidden_states.dtype))

    def apply_es_update(
        self,
        *,
        utilities: torch.Tensor,
        noise_bundle: dict[str, torch.Tensor],
        alpha_config: Any,
        sigma_config: Any,
    ) -> None:
        raise RuntimeError("legacy ES update path has been removed; use payload-based updates instead")

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

    def export_trainable_state(self) -> dict[str, torch.Tensor]:
        return {
            "m_state": self.m_state.detach().cpu().clone(),
            "effective_m_state": torch.diag(self.m_state.detach()).cpu(),
        }

    def load_trainable_state(self, payload: dict[str, torch.Tensor]) -> None:
        state = payload.get("m_state")
        if state is None:
            raise KeyError(f"{self.layer_name} checkpoint payload is missing m_state")
        state = state.to(self.m_state.device, dtype=self.m_state.dtype)
        if state.ndim == 2:
            if state.shape[0] != state.shape[1] or state.shape[0] != self.m_state.shape[0]:
                raise ValueError(
                    f"{self.layer_name} diagonal state shape mismatch: "
                    f"expected {(self.m_state.shape[0], self.m_state.shape[0])}, got {tuple(state.shape)}"
                )
            state = torch.diagonal(state, dim1=0, dim2=1)
        self.m_state.copy_(state)

    def export_lora_weights(self, *, active_index: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        diag_entries = self.m_state if active_index is None else self.active_m[active_index]
        sqrt_abs = torch.sqrt(diag_entries.float().abs())
        signed_left = torch.sign(diag_entries.float()) * sqrt_abs
        lora_b = (self.u_basis.float() * signed_left.unsqueeze(0)).contiguous()
        lora_a = (sqrt_abs.unsqueeze(1) * self.vh_basis.float()).contiguous()
        return lora_a, lora_b

    def export_lora_rank(self) -> int:
        return int(self.m_state.shape[0])

    def effective_matrix(self) -> torch.Tensor:
        return torch.diag(self.m_state.detach()).cpu()

    def initial_noise_scale(self) -> torch.Tensor | None:
        return self.noise_scale.detach().cpu().clone()


class FactorizedSpectralAdapterLayer(AdapterLayerBase):
    def __init__(
        self,
        *,
        layer_name: str,
        u: torch.Tensor,
        vh: torch.Tensor,
        factor_rank: int,
        init_scale: float = 0.01,
    ) -> None:
        super().__init__(layer_name=layer_name)
        self.register_buffer("u_basis", u.float().contiguous())
        self.register_buffer("vh_basis", vh.float().contiguous())
        self.register_buffer("v_basis", vh.float().transpose(0, 1).contiguous())
        basis_rank = int(vh.shape[0])
        effective_factor_rank = max(1, min(int(factor_rank), basis_rank))
        state = torch.zeros((2, basis_rank, effective_factor_rank), dtype=torch.float32, device=u.device)
        if init_scale > 0:
            state[0].normal_(mean=0.0, std=float(init_scale))
        self.register_buffer("m_state", state)
        self.active_state: torch.Tensor | None = None

    def _split_state(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return state[0], state[1]

    def _materialize_matrix(self, state: torch.Tensor) -> torch.Tensor:
        left, right = self._split_state(state)
        return left.float() @ right.float().transpose(0, 1)

    def sample_noise(self, num_mutants: int, *, antithetic: bool = True) -> dict[str, torch.Tensor]:
        return {"m": _sample_noise_like(self.m_state, num_mutants=num_mutants, antithetic=antithetic)}

    def activate_mutants(self, noise_bundle: dict[str, Any], sigma_config: Any) -> None:
        sigma = _resolve_sigma(noise_bundle, sigma_config)
        self.active_state = self.m_state.unsqueeze(0) + sigma * noise_bundle["m"]

    def activate_current(self) -> None:
        self.active_state = self.m_state.unsqueeze(0)

    def clear_active(self) -> None:
        self.active_state = None

    def forward_delta(self, hidden_states: torch.Tensor, mutant_indices: torch.Tensor) -> torch.Tensor:
        if self.active_state is None:
            return torch.zeros(
                (*hidden_states.shape[:-1], self.u_basis.shape[0]),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        states = self.active_state[mutant_indices].to(hidden_states.dtype)
        left = states[:, 0]
        right = states[:, 1]
        proj = torch.einsum("bsi,ir->bsr", hidden_states, self.v_basis.to(hidden_states.dtype))
        latent = torch.einsum("bsr,brk->bsk", proj, right)
        mixed = torch.einsum("bsk,brk->bsr", latent, left)
        return torch.einsum("bsr,or->bso", mixed, self.u_basis.to(hidden_states.dtype))

    def apply_es_update(
        self,
        *,
        utilities: torch.Tensor,
        noise_bundle: dict[str, torch.Tensor],
        alpha_config: Any,
        sigma_config: Any,
    ) -> None:
        raise RuntimeError("legacy ES update path has been removed; use payload-based updates instead")

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

    def export_trainable_state(self) -> dict[str, torch.Tensor]:
        left, right = self._split_state(self.m_state)
        return {
            "m_state": self.m_state.detach().cpu().clone(),
            "left_state": left.detach().cpu().clone(),
            "right_state": right.detach().cpu().clone(),
            "effective_m_state": self._materialize_matrix(self.m_state).cpu(),
        }

    def load_trainable_state(self, payload: dict[str, torch.Tensor]) -> None:
        if "m_state" in payload:
            state = payload["m_state"].to(self.m_state.device, dtype=self.m_state.dtype)
            if tuple(state.shape) == tuple(self.m_state.shape):
                self.m_state.copy_(state)
                return
        left = payload.get("left_state")
        right = payload.get("right_state")
        if left is None or right is None:
            raise KeyError(f"{self.layer_name} checkpoint payload is missing factorized state tensors")
        packed = torch.stack(
            [
                left.to(self.m_state.device, dtype=self.m_state.dtype),
                right.to(self.m_state.device, dtype=self.m_state.dtype),
            ],
            dim=0,
        )
        self.m_state.copy_(packed)

    def export_lora_weights(self, *, active_index: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.m_state if active_index is None else self.active_state[active_index]
        left, right = self._split_state(state)
        lora_b = (self.u_basis.float() @ left.float()).contiguous()
        lora_a = (right.float().transpose(0, 1) @ self.vh_basis.float()).contiguous()
        return lora_a, lora_b

    def export_lora_rank(self) -> int:
        return int(self.m_state.shape[2])

    def effective_matrix(self) -> torch.Tensor:
        return self._materialize_matrix(self.m_state).cpu()


class LoRAESAdapterLayer(AdapterLayerBase):
    def __init__(
        self,
        *,
        layer_name: str,
        out_dim: int,
        in_dim: int,
        factor_rank: int,
        init_scale: float = 0.01,
    ) -> None:
        super().__init__(layer_name=layer_name)
        effective_factor_rank = max(1, min(int(factor_rank), int(min(out_dim, in_dim))))
        state = torch.zeros((out_dim + in_dim, effective_factor_rank), dtype=torch.float32)
        if init_scale > 0:
            state[out_dim:, :].normal_(mean=0.0, std=float(init_scale))
        self.register_buffer("m_state", state)
        self.out_dim = int(out_dim)
        self.in_dim = int(in_dim)
        self.active_state: torch.Tensor | None = None

    def _split_state(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lora_b = state[: self.out_dim]
        lora_a_t = state[self.out_dim :]
        return lora_b, lora_a_t

    def sample_noise(self, num_mutants: int, *, antithetic: bool = True) -> dict[str, torch.Tensor]:
        return {"m": _sample_noise_like(self.m_state, num_mutants=num_mutants, antithetic=antithetic)}

    def activate_mutants(self, noise_bundle: dict[str, Any], sigma_config: Any) -> None:
        sigma = _resolve_sigma(noise_bundle, sigma_config)
        self.active_state = self.m_state.unsqueeze(0) + sigma * noise_bundle["m"]

    def activate_current(self) -> None:
        self.active_state = self.m_state.unsqueeze(0)

    def clear_active(self) -> None:
        self.active_state = None

    def forward_delta(self, hidden_states: torch.Tensor, mutant_indices: torch.Tensor) -> torch.Tensor:
        if self.active_state is None:
            return torch.zeros(
                (*hidden_states.shape[:-1], self.out_dim),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        states = self.active_state[mutant_indices].to(hidden_states.dtype)
        lora_b = states[:, : self.out_dim]
        lora_a_t = states[:, self.out_dim :]
        latent = torch.einsum("bsi,bik->bsk", hidden_states, lora_a_t)
        return torch.einsum("bsk,bok->bso", latent, lora_b)

    def apply_es_update(
        self,
        *,
        utilities: torch.Tensor,
        noise_bundle: dict[str, torch.Tensor],
        alpha_config: Any,
        sigma_config: Any,
    ) -> None:
        raise RuntimeError("legacy ES update path has been removed; use payload-based updates instead")

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

    def export_trainable_state(self) -> dict[str, torch.Tensor]:
        lora_b, lora_a_t = self._split_state(self.m_state)
        return {
            "m_state": self.m_state.detach().cpu().clone(),
            "lora_b_state": lora_b.detach().cpu().clone(),
            "lora_a_state": lora_a_t.transpose(0, 1).detach().cpu().clone(),
        }

    def load_trainable_state(self, payload: dict[str, torch.Tensor]) -> None:
        if "m_state" in payload:
            state = payload["m_state"].to(self.m_state.device, dtype=self.m_state.dtype)
            if tuple(state.shape) == tuple(self.m_state.shape):
                self.m_state.copy_(state)
                return
        lora_b = payload.get("lora_b_state")
        lora_a = payload.get("lora_a_state")
        if lora_b is None or lora_a is None:
            raise KeyError(f"{self.layer_name} checkpoint payload is missing LoRA trainable states")
        packed = torch.cat(
            [
                lora_b.to(self.m_state.device, dtype=self.m_state.dtype),
                lora_a.to(self.m_state.device, dtype=self.m_state.dtype).transpose(0, 1),
            ],
            dim=0,
        )
        self.m_state.copy_(packed)

    def export_lora_weights(self, *, active_index: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.m_state if active_index is None else self.active_state[active_index]
        lora_b, lora_a_t = self._split_state(state)
        return lora_a_t.float().transpose(0, 1).contiguous(), lora_b.float().contiguous()

    def export_lora_rank(self) -> int:
        return int(self.m_state.shape[1])

    def effective_matrix(self) -> torch.Tensor | None:
        return None
