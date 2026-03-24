from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from config_utils import apply_overrides, load_yaml_config


def _write_model_config(model_dir: Path, *, num_hidden_layers: int) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "config.json").open("w", encoding="utf-8") as handle:
        handle.write(f'{{"num_hidden_layers": {num_hidden_layers}}}')


def _write_train_config(config_path: Path, *, model_path: Path, target_blocks: object) -> None:
    payload = {
        "model": {"model_path": str(model_path)},
        "layers": {"target_blocks": target_blocks},
    }
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def test_load_yaml_config_expands_all_blocks(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    config_path = tmp_path / "config.yaml"
    _write_model_config(model_dir, num_hidden_layers=36)
    _write_train_config(config_path, model_path=model_dir, target_blocks="all-blocks")

    config = load_yaml_config(config_path)

    assert config["layers"]["target_blocks"] == list(range(36))


def test_apply_overrides_accepts_target_block_count(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    config_path = tmp_path / "config.yaml"
    _write_model_config(model_dir, num_hidden_layers=36)
    _write_train_config(config_path, model_path=model_dir, target_blocks=[0, 1, 2])

    config = apply_overrides(
        load_yaml_config(config_path),
        [
            "layers.target_blocks=28",
        ],
    )

    assert config["layers"]["target_blocks"] == list(range(28))


def test_apply_overrides_preserves_explicit_block_list(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    config_path = tmp_path / "config.yaml"
    _write_model_config(model_dir, num_hidden_layers=36)
    _write_train_config(config_path, model_path=model_dir, target_blocks=[0, 2, 4])

    config = apply_overrides(load_yaml_config(config_path), ["layers.target_blocks=[1,3,5]"])

    assert config["layers"]["target_blocks"] == [1, 3, 5]


def test_apply_overrides_rejects_out_of_range_block_count(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    config_path = tmp_path / "config.yaml"
    _write_model_config(model_dir, num_hidden_layers=8)
    _write_train_config(config_path, model_path=model_dir, target_blocks=[0, 1, 2])

    with pytest.raises(ValueError, match="exceeds model block count"):
        apply_overrides(load_yaml_config(config_path), ["layers.target_blocks=9"])
