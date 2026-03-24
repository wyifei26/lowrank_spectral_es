import torch.nn as nn
import pytest

from models.layer_selector import LayerSelection
from models.svd_cache import (
    _band_indices,
    _resolve_effective_rank,
    create_svd_cache,
    load_or_create_svd_cache,
    resolve_svd_cache_path,
)


def test_band_indices_lengths():
    assert _band_indices(32, 8, "top-band") == list(range(8))
    assert len(_band_indices(32, 8, "middle-band")) == 8
    assert len(_band_indices(32, 8, "mixed-band")) == 8


def test_rank_zero_uses_full_rank():
    assert _resolve_effective_rank(32, 0) == 32
    assert _band_indices(32, 0, "top-band") == list(range(32))
    assert len(_band_indices(32, 0, "middle-band")) == 32
    assert len(_band_indices(32, 0, "mixed-band")) == 32


def test_negative_rank_is_rejected():
    with pytest.raises(ValueError):
        _resolve_effective_rank(32, -1)


def test_load_or_create_svd_cache_materializes_and_reuses(tmp_path):
    linear = nn.Linear(6, 4, bias=False)
    selection = LayerSelection(
        full_name="model.layers.0.self_attn.q_proj",
        module_key="q_proj",
        block_index=0,
        parent_module=nn.Module(),
        child_name="q_proj",
        module=linear,
    )
    cache_path = resolve_svd_cache_path(
        model_path="/tmp/fake-model",
        selections=[selection],
        rank=2,
        band_strategy="top-band",
        cache_dir=tmp_path,
    )

    layers = create_svd_cache(
        cache_path=cache_path,
        selections=[selection],
        rank=2,
        band_strategy="top-band",
        model_path="/tmp/fake-model",
        device="cpu",
    )
    assert cache_path.exists()
    assert layers[selection.full_name]["u"].shape == (4, 2)

    cache_path_second, layers_second = load_or_create_svd_cache(
        model_path="/tmp/fake-model",
        selections=[selection],
        rank=2,
        band_strategy="top-band",
        cache_dir=tmp_path,
        device="cpu",
    )
    assert cache_path_second == cache_path
    assert layers_second[selection.full_name]["vh"].shape == (2, 6)
