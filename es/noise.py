import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_antithetic_normal(
    *,
    shape: tuple[int, ...],
    num_mutants: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if num_mutants % 2 != 0:
        raise ValueError("antithetic sampling requires an even num_mutants")
    half = num_mutants // 2
    base = torch.randn((half, *shape), device=device, dtype=dtype)
    return torch.cat([base, -base], dim=0)


def sample_standard_normal(
    *,
    shape: tuple[int, ...],
    num_mutants: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if num_mutants <= 0:
        raise ValueError("num_mutants must be positive")
    return torch.randn((num_mutants, *shape), device=device, dtype=dtype)
