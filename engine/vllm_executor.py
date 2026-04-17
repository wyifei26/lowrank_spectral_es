from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from vllm import SamplingParams
from vllm.lora.request import LoRARequest

from data.common import canonical_benchmark_name
from engine.profiler import ThroughputProfiler
from eval.reward_router import score_record_prediction
from models.spectral_vllm import SpectralVLLMState


@dataclass
class VLLMBatchExecutionResult:
    rewards: torch.Tensor
    exact_match_rates: torch.Tensor
    predictions: list[dict]
    profiler_snapshot: dict[str, float]
    benchmark_metrics: dict[str, dict[str, float]]


class VLLMSpectralExecutor:
    def __init__(
        self,
        *,
        llm,
        state: SpectralVLLMState,
        model_path: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        top_k: int = -1,
        presence_penalty: float = 0.0,
        mutant_chunk_size: int,
        adapter_root: str | Path,
        rank: int = 0,
        reward_config: dict[str, Any] | None = None,
    ) -> None:
        self.llm = llm
        self.state = state
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.mutant_chunk_size = mutant_chunk_size
        self.adapter_root = Path(adapter_root)
        self.adapter_root.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.reward_config = reward_config or {}
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            max_tokens=max_new_tokens,
        )
        self._request_counter = 1

    def _next_lora_request(self, *, path: Path, slot_name: str) -> LoRARequest:
        request = LoRARequest(
            lora_name=f"{slot_name}_{self._request_counter}",
            lora_int_id=self._request_counter,
            lora_path=str(path),
        )
        self._request_counter += 1
        return request

    def _extract_text_and_tokens(self, outputs) -> tuple[list[str], int, list[int]]:
        texts: list[str] = []
        token_count = 0
        token_lengths: list[int] = []
        for item in outputs:
            if not item.outputs:
                texts.append("")
                token_lengths.append(0)
                continue
            best = item.outputs[0]
            texts.append(best.text)
            token_ids = getattr(best, "token_ids", None)
            if token_ids is not None:
                token_count += len(token_ids)
                token_lengths.append(len(token_ids))
            else:
                token_lengths.append(0)
        return texts, token_count, token_lengths

    def score_current_state(
        self,
        *,
        records: list[dict],
        question_micro_batch: int,
        collect_predictions: bool = False,
        use_base_model: bool = False,
    ) -> VLLMBatchExecutionResult:
        profiler = ThroughputProfiler()
        predictions: list[dict] = []
        rewards = torch.zeros(1, dtype=torch.float32)
        counts = torch.zeros(1, dtype=torch.float32)
        exact_matches = torch.zeros(1, dtype=torch.float32)
        benchmark_metrics: dict[str, dict[str, float]] = {}

        current_request = None
        if not use_base_model:
            current_state_name = f"current_state_rank{self.rank:02d}"
            current_dir = self.state.export_adapter(
                output_dir=self.adapter_root / current_state_name,
                base_model_name_or_path=self.model_path,
                adapter_name=current_state_name,
            )
            current_request = self._next_lora_request(path=current_dir, slot_name=current_state_name)

        for q_start in range(0, len(records), question_micro_batch):
            q_records = records[q_start : q_start + question_micro_batch]
            prompts = [item["prompt"] for item in q_records]
            generate_kwargs = {
                "prompts": prompts,
                "sampling_params": self.sampling_params,
                "use_tqdm": False,
            }
            if current_request is not None:
                generate_kwargs["lora_request"] = [current_request] * len(prompts)
            outputs = self.llm.generate(
                **generate_kwargs,
            )
            texts, token_count, token_lengths = self._extract_text_and_tokens(outputs)
            profiler.record(generated_tokens=token_count, requests=len(prompts), mutant_evals=1)
            for idx, text in enumerate(texts):
                result = score_record_prediction(
                    text,
                    q_records[idx],
                    reward_config=self.reward_config,
                )
                rewards[0] += result.reward
                counts[0] += 1
                exact_matches[0] += float(result.correct)
                benchmark_name = canonical_benchmark_name(
                    q_records[idx].get("data_source") or q_records[idx].get("source")
                )
                benchmark_payload = benchmark_metrics.setdefault(
                    benchmark_name,
                    {"num_examples": 0.0, "reward_sum": 0.0, "exact_match_sum": 0.0},
                )
                benchmark_payload["num_examples"] += 1.0
                benchmark_payload["reward_sum"] += float(result.reward)
                benchmark_payload["exact_match_sum"] += float(result.correct)
                if collect_predictions:
                    predictions.append(
                        {
                            "record_index": q_records[idx].get("_record_index"),
                            "id": q_records[idx]["id"],
                            "data_source": q_records[idx].get("data_source"),
                            "benchmark": benchmark_name,
                            "prompt": q_records[idx]["prompt"],
                            "gold_value": result.gold_value,
                            "prediction": text,
                            "reward": result.reward,
                            "predicted_value": result.predicted_value,
                            "correct": result.correct,
                            "generated_tokens": token_lengths[idx],
                            "mutant_index": 0,
                        }
                    )

        counts = counts.clamp_min(1.0)
        return VLLMBatchExecutionResult(
            rewards=rewards / counts,
            exact_match_rates=exact_matches / counts,
            predictions=predictions,
            profiler_snapshot=profiler.snapshot(),
            benchmark_metrics=benchmark_metrics,
        )

    def score_active_mutants(
        self,
        *,
        records: list[dict],
        num_mutants: int,
        question_micro_batch: int,
        collect_predictions: bool = False,
    ) -> VLLMBatchExecutionResult:
        return self.score_mutant_subset(
            records=records,
            mutant_indices=list(range(num_mutants)),
            question_micro_batch=question_micro_batch,
            collect_predictions=collect_predictions,
        )

    def score_mutant_subset(
        self,
        *,
        records: list[dict],
        mutant_indices: list[int],
        question_micro_batch: int,
        collect_predictions: bool = False,
    ) -> VLLMBatchExecutionResult:
        profiler = ThroughputProfiler()
        rewards = torch.zeros(len(mutant_indices), dtype=torch.float32)
        counts = torch.zeros(len(mutant_indices), dtype=torch.float32)
        exact_matches = torch.zeros(len(mutant_indices), dtype=torch.float32)
        predictions: list[dict] = []

        local_index_by_mutant = {mutant_index: local_index for local_index, mutant_index in enumerate(mutant_indices)}

        for m_start in range(0, len(mutant_indices), self.mutant_chunk_size):
            chunk_mutants = mutant_indices[m_start : m_start + self.mutant_chunk_size]
            chunk_requests: list[tuple[int, LoRARequest]] = []
            for slot_index, global_mutant in enumerate(chunk_mutants):
                path = self.state.export_adapter(
                    output_dir=self.adapter_root / f"mutant_{global_mutant:04d}",
                    base_model_name_or_path=self.model_path,
                    adapter_name=f"mutant_{global_mutant:04d}",
                    active_index=global_mutant,
                )
                chunk_requests.append(
                    (
                        global_mutant,
                        self._next_lora_request(path=path, slot_name=f"mutant_{slot_index}"),
                    )
                )

            for q_start in range(0, len(records), question_micro_batch):
                q_records = records[q_start : q_start + question_micro_batch]
                q_prompts = [item["prompt"] for item in q_records]
                flat_prompts: list[str] = []
                flat_requests: list[LoRARequest] = []
                flat_pairs: list[tuple[int, int]] = []
                for global_mutant, request in chunk_requests:
                    flat_prompts.extend(q_prompts)
                    flat_requests.extend([request] * len(q_prompts))
                    flat_pairs.extend([(global_mutant, q_idx) for q_idx in range(len(q_prompts))])

                outputs = self.llm.generate(
                    flat_prompts,
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                    lora_request=flat_requests,
                )
                texts, token_count, token_lengths = self._extract_text_and_tokens(outputs)
                profiler.record(
                    generated_tokens=token_count,
                    requests=len(flat_prompts),
                    mutant_evals=len(chunk_mutants),
                )

                for local_row, text in enumerate(texts):
                    global_mutant, question_idx = flat_pairs[local_row]
                    local_mutant = local_index_by_mutant[global_mutant]
                    record = q_records[question_idx]
                    result = score_record_prediction(
                        text,
                        record,
                        reward_config=self.reward_config,
                    )
                    rewards[local_mutant] += result.reward
                    counts[local_mutant] += 1
                    exact_matches[local_mutant] += float(result.correct)
                    if collect_predictions:
                        predictions.append(
                            {
                                "record_index": record.get("_record_index"),
                                "id": record["id"],
                                "prompt": record["prompt"],
                                "gold_value": result.gold_value,
                                "prediction": text,
                                "reward": result.reward,
                                "predicted_value": result.predicted_value,
                                "correct": result.correct,
                                "generated_tokens": token_lengths[local_row],
                                "mutant_index": global_mutant,
                            }
                        )

        counts = counts.clamp_min(1.0)
        return VLLMBatchExecutionResult(
            rewards=rewards / counts,
            exact_match_rates=exact_matches / counts,
            predictions=predictions,
            profiler_snapshot=profiler.snapshot(),
            benchmark_metrics={},
        )
