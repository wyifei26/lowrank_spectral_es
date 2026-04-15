from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent


def load_raw_yaml_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"expected dict config in {path}, got {type(payload)}")
    return payload


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    payload = load_raw_yaml_config(path)
    return normalize_config(payload)


def dump_yaml_config(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _parse_override_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except Exception:
        return raw


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    updated = deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got: {item}")
        key, raw_value = item.split("=", 1)
        value = _parse_override_value(raw_value)
        target = updated
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value
    return normalize_config(updated)


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(config)
    _normalize_data_dirs(normalized)
    _normalize_layer_blocks(normalized)
    return normalized


def _normalize_data_dirs(config: dict[str, Any]) -> None:
    data = config.get("data")
    if not isinstance(data, dict):
        return

    source = str(data.get("source") or "gsm8k").strip().replace("-", "_")
    default_root = PROJECT_ROOT / "dataset" / source

    root_dir_raw = data.get("root_dir")
    raw_dir_raw = data.get("raw_dir") or data.get("cache_dir")
    processed_dir_raw = data.get("processed_dir")
    processed_exports_dir_raw = data.get("processed_exports_dir")
    manifest_path_raw = data.get("manifest_path")

    if root_dir_raw:
        root_dir = Path(root_dir_raw)
    elif raw_dir_raw:
        root_dir = Path(raw_dir_raw).parent
    elif processed_dir_raw:
        root_dir = Path(processed_dir_raw).parent
    else:
        root_dir = default_root

    raw_dir = Path(raw_dir_raw) if raw_dir_raw else root_dir / "raw"
    processed_dir = Path(processed_dir_raw) if processed_dir_raw else root_dir / "processed"
    processed_exports_dir = (
        Path(processed_exports_dir_raw) if processed_exports_dir_raw else root_dir / "processed_exports"
    )
    manifest_path = Path(manifest_path_raw) if manifest_path_raw else root_dir / "manifest.json"

    data["root_dir"] = str(root_dir)
    data["raw_dir"] = str(raw_dir)
    data["cache_dir"] = str(raw_dir)
    data["processed_dir"] = str(processed_dir)
    data["processed_exports_dir"] = str(processed_exports_dir)
    data["manifest_path"] = str(manifest_path)


def _normalize_layer_blocks(config: dict[str, Any]) -> None:
    layers = config.get("layers")
    if not isinstance(layers, dict) or "target_blocks" not in layers:
        return
    total_blocks = _resolve_total_transformer_blocks(config)
    layers["target_blocks"] = _resolve_target_blocks(layers["target_blocks"], total_blocks=total_blocks)


def _resolve_total_transformer_blocks(config: dict[str, Any]) -> int:
    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError("config.model is required to resolve layers.target_blocks")
    model_path = model.get("model_path")
    if not isinstance(model_path, str) or not model_path:
        raise ValueError("config.model.model_path is required to resolve layers.target_blocks")

    config_path = Path(model_path) / "config.json"
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        num_hidden_layers = payload.get("num_hidden_layers")
        if isinstance(num_hidden_layers, int) and num_hidden_layers > 0:
            return num_hidden_layers

    try:
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as exc:
        raise ValueError(
            f"could not resolve num_hidden_layers from model path {model_path}"
        ) from exc
    num_hidden_layers = getattr(hf_config, "num_hidden_layers", None)
    if not isinstance(num_hidden_layers, int) or num_hidden_layers <= 0:
        raise ValueError(f"model at {model_path} does not expose a valid num_hidden_layers")
    return num_hidden_layers


def _resolve_target_blocks(raw_target_blocks: Any, *, total_blocks: int) -> list[int]:
    if isinstance(raw_target_blocks, str):
        if raw_target_blocks != "all-blocks":
            raise ValueError(
                "layers.target_blocks must be a block index list, a positive integer, or 'all-blocks'"
            )
        return list(range(total_blocks))

    if isinstance(raw_target_blocks, int):
        if raw_target_blocks <= 0:
            raise ValueError("layers.target_blocks must be positive when provided as a block count")
        if raw_target_blocks > total_blocks:
            raise ValueError(
                f"layers.target_blocks={raw_target_blocks} exceeds model block count {total_blocks}"
            )
        return list(range(raw_target_blocks))

    if not isinstance(raw_target_blocks, list) or not raw_target_blocks:
        raise ValueError(
            "layers.target_blocks must be a non-empty block index list, a positive integer, or 'all-blocks'"
        )

    normalized_blocks: list[int] = []
    seen_blocks: set[int] = set()
    for block_index in raw_target_blocks:
        if not isinstance(block_index, int):
            raise TypeError(f"layers.target_blocks entries must be integers, got {type(block_index)}")
        if block_index < 0 or block_index >= total_blocks:
            raise ValueError(
                f"layers.target_blocks contains out-of-range block {block_index}; model has {total_blocks} blocks"
            )
        if block_index in seen_blocks:
            raise ValueError(f"layers.target_blocks contains duplicate block index {block_index}")
        seen_blocks.add(block_index)
        normalized_blocks.append(block_index)
    return normalized_blocks
