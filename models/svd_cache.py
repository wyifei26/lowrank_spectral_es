import hashlib
from pathlib import Path

import torch

from models.layer_selector import LayerSelection


def _resolve_effective_rank(num_singular: int, requested_rank: int) -> int:
    if requested_rank < 0:
        raise ValueError(f"rank must be >= 0, got {requested_rank}")
    if requested_rank == 0:
        return num_singular
    return min(requested_rank, num_singular)


def _band_indices(num_singular: int, rank: int, band_strategy: str) -> list[int]:
    rank = _resolve_effective_rank(num_singular, rank)
    if band_strategy == "top-band":
        return list(range(rank))
    if band_strategy == "middle-band":
        start = max(0, (num_singular // 2) - (rank // 2))
        end = min(num_singular, start + rank)
        start = max(0, end - rank)
        return list(range(start, end))
    if band_strategy == "mixed-band":
        top_count = rank // 2
        middle_count = rank - top_count
        top = list(range(top_count))
        middle_start = max(0, (num_singular // 2) - (middle_count // 2))
        middle = list(range(middle_start, min(num_singular, middle_start + middle_count)))
        merged = []
        seen = set()
        for idx in top + middle:
            if idx not in seen:
                merged.append(idx)
                seen.add(idx)
        candidate = 0
        while len(merged) < rank and candidate < num_singular:
            if candidate not in seen:
                merged.append(candidate)
                seen.add(candidate)
            candidate += 1
        return merged[:rank]
    raise KeyError(f"unsupported band strategy: {band_strategy}")


def _cache_file_name(model_path: str, rank: int, band_strategy: str, layer_names: list[str]) -> str:
    rank_tag = "full" if rank == 0 else str(rank)
    signature = "|".join([model_path, rank_tag, band_strategy, *layer_names])
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12]
    return f"svd_cache_r{rank_tag}_{band_strategy}_{digest}.pt"


def resolve_svd_cache_path(
    *,
    model_path: str,
    selections: list[LayerSelection],
    rank: int,
    band_strategy: str,
    cache_dir: str | Path,
) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / _cache_file_name(model_path, rank, band_strategy, [item.full_name for item in selections])


def create_svd_cache(
    *,
    cache_path: str | Path,
    selections: list[LayerSelection],
    rank: int,
    band_strategy: str,
    model_path: str,
    device: torch.device | str = "cpu",
    compute_dtype: torch.dtype = torch.float32,
) -> dict[str, dict[str, torch.Tensor]]:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if rank < 0:
        raise ValueError(f"rank must be >= 0, got {rank}")
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        return payload["layers"]

    layers: dict[str, dict[str, torch.Tensor]] = {}
    resolved_max_rank = 0
    target_device = torch.device(device)
    for selection in selections:
        weight = selection.module.weight.detach().to(device=target_device, dtype=compute_dtype)
        u, s, vh = torch.linalg.svd(weight, full_matrices=False)
        indices = _band_indices(len(s), rank, band_strategy)
        resolved_max_rank = max(resolved_max_rank, len(indices))
        layers[selection.full_name] = {
            "indices": torch.tensor(indices, dtype=torch.long),
            "u": u[:, indices].contiguous().cpu(),
            "s": s[indices].contiguous().cpu(),
            "vh": vh[indices, :].contiguous().cpu(),
        }
        del weight, u, s, vh
        if target_device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "model_path": model_path,
        "requested_rank": rank,
        "rank": resolved_max_rank,
        "band_strategy": band_strategy,
        "layers": layers,
    }
    tmp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(cache_path)
    return layers


def load_or_create_svd_cache(
    *,
    model_path: str,
    selections: list[LayerSelection],
    rank: int,
    band_strategy: str,
    cache_dir: str | Path,
    device: torch.device | str = "cpu",
    compute_dtype: torch.dtype = torch.float32,
) -> tuple[Path, dict[str, dict[str, torch.Tensor]]]:
    cache_path = resolve_svd_cache_path(
        model_path=model_path,
        selections=selections,
        rank=rank,
        band_strategy=band_strategy,
        cache_dir=cache_dir,
    )
    layers = create_svd_cache(
        cache_path=cache_path,
        selections=selections,
        rank=rank,
        band_strategy=band_strategy,
        model_path=model_path,
        device=device,
        compute_dtype=compute_dtype,
    )
    return cache_path, layers
