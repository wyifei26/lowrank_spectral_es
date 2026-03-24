from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from engine.gpu_monitor import GPUMonitorSnapshot


@dataclass(frozen=True)
class MutantShard:
    rank: int
    world_size: int
    start: int
    end: int

    @property
    def count(self) -> int:
        return self.end - self.start

    @property
    def indices(self) -> list[int]:
        return list(range(self.start, self.end))


def resolve_mutant_shard(*, num_mutants: int, world_size: int, rank: int, mutants_per_worker: int) -> MutantShard:
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    if mutants_per_worker <= 0:
        raise ValueError(f"mutants_per_worker must be positive, got {mutants_per_worker}")
    expected_mutants = world_size * mutants_per_worker
    if expected_mutants != num_mutants:
        raise ValueError(
            f"num_mutants must equal world_size * mutants_per_worker, got "
            f"{num_mutants} vs {world_size} * {mutants_per_worker}"
        )
    start = rank * mutants_per_worker
    end = start + mutants_per_worker
    return MutantShard(rank=rank, world_size=world_size, start=start, end=end)


def merge_sharded_rewards(
    *,
    num_mutants: int,
    shards: list[MutantShard],
    shard_rewards: list[list[float]],
) -> torch.Tensor:
    if len(shards) != len(shard_rewards):
        raise ValueError("shards and shard_rewards must have the same length")
    rewards = torch.empty(num_mutants, dtype=torch.float32)
    for shard, local_rewards in zip(shards, shard_rewards):
        if len(local_rewards) != shard.count:
            raise ValueError(
                f"shard reward length mismatch for rank {shard.rank}: "
                f"expected {shard.count}, got {len(local_rewards)}"
            )
        rewards[shard.start : shard.end] = torch.tensor(local_rewards, dtype=torch.float32)
    return rewards


def aggregate_distributed_step_metrics(
    *,
    profiler_snapshots: list[dict[str, float]],
    gpu_snapshots: list[GPUMonitorSnapshot],
    step_durations: list[float],
) -> dict[str, Any]:
    if not profiler_snapshots:
        raise ValueError("profiler_snapshots must be non-empty")
    if len(profiler_snapshots) != len(gpu_snapshots) or len(profiler_snapshots) != len(step_durations):
        raise ValueError("profiler_snapshots, gpu_snapshots, and step_durations must have the same length")

    elapsed = max(
        max(float(item.get("elapsed_seconds", 0.0)) for item in profiler_snapshots),
        max(step_durations),
        1e-6,
    )
    total_tokens = sum(float(item.get("generated_tokens_total", 0.0)) for item in profiler_snapshots)
    total_requests = sum(float(item.get("requests_total", 0.0)) for item in profiler_snapshots)
    total_mutants = sum(float(item.get("mutant_evals_total", 0.0)) for item in profiler_snapshots)
    per_rank_gpu_util_mean = [float(item.gpu_util_mean) for item in gpu_snapshots]
    per_rank_gpu_util_max = [float(item.gpu_util_max) for item in gpu_snapshots]
    per_rank_mem_util_mean = [float(item.mem_util_mean) for item in gpu_snapshots]
    per_rank_mem_util_max = [float(item.mem_util_max) for item in gpu_snapshots]
    per_rank_mem_used_gb_mean = [float(item.mem_used_gb_mean) for item in gpu_snapshots]
    per_rank_mem_used_gb_max = [float(item.mem_used_gb_max) for item in gpu_snapshots]
    per_rank_monitor_samples = [float(item.samples) for item in gpu_snapshots]

    return {
        "elapsed_seconds": elapsed,
        "step_wall_time_seconds": elapsed,
        "generated_tokens_total": total_tokens,
        "requests_total": total_requests,
        "mutant_evals_total": total_mutants,
        "tokens_per_sec": total_tokens / elapsed,
        "requests_per_sec": total_requests / elapsed,
        "mutants_per_sec": total_mutants / elapsed,
        "gpu_monitor_samples": float(sum(per_rank_monitor_samples)),
        "gpu_util_mean": sum(per_rank_gpu_util_mean) / max(len(per_rank_gpu_util_mean), 1),
        "gpu_util_max": max(per_rank_gpu_util_max, default=0.0),
        "mem_util_mean": sum(per_rank_mem_util_mean) / max(len(per_rank_mem_util_mean), 1),
        "mem_util_max": max(per_rank_mem_util_max, default=0.0),
        "mem_used_gb_mean": sum(per_rank_mem_used_gb_mean) / max(len(per_rank_mem_used_gb_mean), 1),
        "mem_used_gb_max": max(per_rank_mem_used_gb_max, default=0.0),
        "per_rank_gpu_monitor": [asdict(item) for item in gpu_snapshots],
    }
