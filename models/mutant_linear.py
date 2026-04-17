from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class AdapterLayerBase(nn.Module, ABC):
    def __init__(self, layer_name: str):
        super().__init__()
        self.layer_name = layer_name

    @abstractmethod
    def sample_noise(self, num_mutants: int, *, antithetic: bool = True) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def activate_mutants(self, noise_bundle: dict[str, Any], sigma_config: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def activate_current(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def clear_active(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward_delta(self, hidden_states: torch.Tensor, mutant_indices: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def apply_es_update(
        self,
        *,
        utilities: torch.Tensor,
        noise_bundle: dict[str, torch.Tensor],
        alpha_config: Any,
        sigma_config: Any,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def state_norm(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def apply_step_payload(
        self,
        step_bundle: dict[str, torch.Tensor],
        *,
        max_state_norm: float | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def export_trainable_state(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def load_trainable_state(self, payload: dict[str, torch.Tensor]) -> None:
        raise NotImplementedError

    @abstractmethod
    def export_lora_weights(self, *, active_index: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def export_lora_rank(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def effective_matrix(self) -> torch.Tensor | None:
        raise NotImplementedError

    def initial_noise_scale(self) -> torch.Tensor | None:
        return None


class MutantLinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, adapter: AdapterLayerBase):
        super().__init__()
        self.base_layer = base_layer
        self.adapter = adapter
        self.current_mutant_indices: torch.Tensor | None = None
        for param in self.base_layer.parameters():
            param.requires_grad_(False)

    def set_mutant_indices(self, mutant_indices: torch.Tensor | None) -> None:
        self.current_mutant_indices = mutant_indices

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(hidden_states)
        if self.current_mutant_indices is None:
            return base_output
        return base_output + self.adapter.forward_delta(hidden_states, self.current_mutant_indices)


class MutantModel(nn.Module):
    def __init__(self, model: nn.Module, mutant_modules: dict[str, MutantLinear]):
        super().__init__()
        self.model = model
        self.mutant_modules = mutant_modules

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def config(self):
        return self.model.config

    def forward(self, *, mutant_indices: torch.Tensor | None = None, **kwargs):
        if mutant_indices is not None:
            mutant_indices = mutant_indices.to(self.device)
        for module in self.mutant_modules.values():
            module.set_mutant_indices(mutant_indices)
        return self.model(**kwargs)

    def activate_current_state(self) -> None:
        for module in self.mutant_modules.values():
            module.adapter.activate_current()

    def activate_mutants(self, noise_payloads: dict[str, dict[str, torch.Tensor]], sigma_config: Any) -> None:
        for name, module in self.mutant_modules.items():
            module.adapter.activate_mutants(noise_payloads[name], sigma_config)

    def clear_active(self) -> None:
        for module in self.mutant_modules.values():
            module.adapter.clear_active()
            module.set_mutant_indices(None)

    def sample_noise(self, num_mutants: int, *, antithetic: bool = True) -> dict[str, dict[str, torch.Tensor]]:
        return {
            name: module.adapter.sample_noise(num_mutants, antithetic=antithetic)
            for name, module in self.mutant_modules.items()
        }

    def apply_es_update(
        self,
        *,
        utilities: torch.Tensor,
        noise_payloads: dict[str, dict[str, torch.Tensor]],
        alpha_config: Any,
        sigma_config: Any,
    ) -> None:
        for name, module in self.mutant_modules.items():
            module.adapter.apply_es_update(
                utilities=utilities,
                noise_bundle=noise_payloads[name],
                alpha_config=alpha_config,
                sigma_config=sigma_config,
            )

    def adapter_state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        return {name: module.adapter.state_dict() for name, module in self.mutant_modules.items()}

    def load_adapter_state_dict(self, payload: dict[str, dict[str, torch.Tensor]]) -> None:
        for name, state in payload.items():
            self.mutant_modules[name].adapter.load_state_dict(state, strict=True)

    def adapter_norms(self) -> dict[str, float]:
        return {name: module.adapter.state_norm() for name, module in self.mutant_modules.items()}

    def apply_step_payloads(
        self,
        step_payloads: dict[str, dict[str, torch.Tensor]],
        *,
        max_state_norm: float | None = None,
    ) -> None:
        for name, module in self.mutant_modules.items():
            bundle = step_payloads.get(name)
            if not bundle:
                continue
            module.adapter.apply_step_payload(bundle, max_state_norm=max_state_norm)
