import torch
import torch.nn as nn

from engine.distributed_utils import (
    aggregate_distributed_step_metrics,
    merge_sharded_rewards,
    resolve_mutant_shard,
)
from engine.gpu_monitor import GPUMonitorSnapshot
from es.spectral_update import (
    apply_alpha_update_to_direction_payloads,
    compute_gaussian_direction_payloads,
    compute_pairwise_direction_payloads,
)
from models.layer_selector import LayerSelection
from models.spectral_vllm import build_vllm_spectral_state


def _build_state() -> tuple:
    torch.manual_seed(0)
    parent = nn.Module()
    parent.proj = nn.Linear(5, 7, bias=False)
    selection = LayerSelection(
        full_name="model.layers.0.self_attn.q_proj",
        module_key="q_proj",
        block_index=0,
        parent_module=parent,
        child_name="proj",
        module=parent.proj,
    )
    cache_payload = {
        selection.full_name: {
            "u": torch.randn(7, 3, dtype=torch.float32),
            "vh": torch.randn(3, 5, dtype=torch.float32),
        }
    }
    state = build_vllm_spectral_state(
        algorithm_name="spectral_es",
        selections=[selection],
        cache_payload=cache_payload,
    )
    return state, selection


def test_resolve_mutant_shard_splits_contiguously():
    shards = [
        resolve_mutant_shard(num_mutants=32, world_size=4, rank=rank, mutants_per_worker=8)
        for rank in range(4)
    ]
    assert [item.start for item in shards] == [0, 8, 16, 24]
    assert [item.end for item in shards] == [8, 16, 24, 32]
    assert shards[2].indices == list(range(16, 24))


def test_distributed_reward_merge_matches_single_process_pairwise_step():
    state_single, _ = _build_state()
    state_distributed, _ = _build_state()

    noise_payloads = state_single.sample_noise(4)
    global_rewards = torch.tensor([1.0, 0.0, 0.5, -0.5], dtype=torch.float32)

    direction_single, _ = compute_pairwise_direction_payloads(
        noise_payloads=noise_payloads,
        rewards=global_rewards,
    )
    step_single, _ = apply_alpha_update_to_direction_payloads(
        direction_payloads=direction_single,
        alpha_config={"m": 0.1},
        sigma_config={"m": 0.5},
        max_layer_step_config={"m": 0.2},
    )
    state_single.apply_step_payloads(step_single, max_state_norm=0.25)

    shards = [
        resolve_mutant_shard(num_mutants=4, world_size=2, rank=rank, mutants_per_worker=2)
        for rank in range(2)
    ]
    merged_rewards = merge_sharded_rewards(
        num_mutants=4,
        shards=shards,
        shard_rewards=[
            global_rewards[shards[0].start : shards[0].end].tolist(),
            global_rewards[shards[1].start : shards[1].end].tolist(),
        ],
    )
    direction_distributed, _ = compute_pairwise_direction_payloads(
        noise_payloads=noise_payloads,
        rewards=merged_rewards,
    )
    step_distributed, _ = apply_alpha_update_to_direction_payloads(
        direction_payloads=direction_distributed,
        alpha_config={"m": 0.1},
        sigma_config={"m": 0.5},
        max_layer_step_config={"m": 0.2},
    )
    state_distributed.apply_step_payloads(step_distributed, max_state_norm=0.25)

    single_state = state_single.adapter_state_dict()
    distributed_state = state_distributed.adapter_state_dict()
    torch.testing.assert_close(merged_rewards, global_rewards)
    torch.testing.assert_close(
        distributed_state["model.layers.0.self_attn.q_proj"]["m_state"],
        single_state["model.layers.0.self_attn.q_proj"]["m_state"],
    )


def test_aggregate_distributed_step_metrics_combines_rank_stats():
    payload = aggregate_distributed_step_metrics(
        profiler_snapshots=[
            {
                "elapsed_seconds": 2.0,
                "generated_tokens_total": 100.0,
                "requests_total": 16.0,
                "mutant_evals_total": 8.0,
            },
            {
                "elapsed_seconds": 1.5,
                "generated_tokens_total": 120.0,
                "requests_total": 16.0,
                "mutant_evals_total": 8.0,
            },
        ],
        gpu_snapshots=[
            GPUMonitorSnapshot(10, 50.0, 80.0, 40.0, 60.0, 12.0, 13.0),
            GPUMonitorSnapshot(8, 60.0, 90.0, 45.0, 65.0, 11.0, 14.0),
        ],
        step_durations=[2.0, 1.5],
    )

    assert payload["step_wall_time_seconds"] == 2.0
    assert payload["generated_tokens_total"] == 220.0
    assert payload["requests_total"] == 32.0
    assert payload["mutant_evals_total"] == 16.0
    assert payload["gpu_util_max"] == 90.0
    assert payload["mem_used_gb_max"] == 14.0
    assert len(payload["per_rank_gpu_monitor"]) == 2


def test_distributed_reward_merge_matches_single_process_gaussian_step():
    state_single, _ = _build_state()
    state_distributed, _ = _build_state()

    noise_payloads = state_single.sample_noise(4, antithetic=False)
    global_rewards = torch.tensor([1.0, 0.0, 0.5, -0.5], dtype=torch.float32)

    direction_single, _ = compute_gaussian_direction_payloads(
        noise_payloads=noise_payloads,
        rewards=global_rewards,
    )
    step_single, _ = apply_alpha_update_to_direction_payloads(
        direction_payloads=direction_single,
        alpha_config={"m": 0.1},
        sigma_config={"m": 0.5},
        max_layer_step_config={"m": 0.2},
    )
    state_single.apply_step_payloads(step_single, max_state_norm=0.25)

    shards = [
        resolve_mutant_shard(num_mutants=4, world_size=2, rank=rank, mutants_per_worker=2)
        for rank in range(2)
    ]
    merged_rewards = merge_sharded_rewards(
        num_mutants=4,
        shards=shards,
        shard_rewards=[
            global_rewards[shards[0].start : shards[0].end].tolist(),
            global_rewards[shards[1].start : shards[1].end].tolist(),
        ],
    )
    direction_distributed, _ = compute_gaussian_direction_payloads(
        noise_payloads=noise_payloads,
        rewards=merged_rewards,
    )
    step_distributed, _ = apply_alpha_update_to_direction_payloads(
        direction_payloads=direction_distributed,
        alpha_config={"m": 0.1},
        sigma_config={"m": 0.5},
        max_layer_step_config={"m": 0.2},
    )
    state_distributed.apply_step_payloads(step_distributed, max_state_norm=0.25)

    single_state = state_single.adapter_state_dict()
    distributed_state = state_distributed.adapter_state_dict()
    torch.testing.assert_close(merged_rewards, global_rewards)
    torch.testing.assert_close(
        distributed_state["model.layers.0.self_attn.q_proj"]["m_state"],
        single_state["model.layers.0.self_attn.q_proj"]["m_state"],
    )
