from __future__ import annotations

from dataclasses import dataclass

import torch

from engine.greedy_decode import greedy_generate_texts
from engine.profiler import ThroughputProfiler
from eval.reward_router import score_record_prediction
from models.mutant_linear import MutantModel


@dataclass
class BatchExecutionResult:
    rewards: torch.Tensor
    exact_match_rates: torch.Tensor
    predictions: list[dict]
    profiler_snapshot: dict[str, float]


class BatchExecutor:
    def __init__(
        self,
        *,
        model: MutantModel,
        tokenizer,
        max_new_tokens: int,
        mutant_chunk_size: int,
        reward_config: dict | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.mutant_chunk_size = mutant_chunk_size
        self.reward_config = reward_config or {}

    def score_active_mutants(
        self,
        *,
        records: list[dict],
        num_mutants: int,
        question_micro_batch: int,
        collect_predictions: bool = False,
    ) -> BatchExecutionResult:
        profiler = ThroughputProfiler()
        rewards = torch.zeros(num_mutants, dtype=torch.float32, device=self.model.device)
        counts = torch.zeros(num_mutants, dtype=torch.float32, device=self.model.device)
        exact_matches = torch.zeros(num_mutants, dtype=torch.float32, device=self.model.device)
        predictions: list[dict] = []

        for q_start in range(0, len(records), question_micro_batch):
            q_records = records[q_start : q_start + question_micro_batch]
            q_prompts = [item["prompt"] for item in q_records]
            q_ids = [item["id"] for item in q_records]
            q_questions = [item["question"] for item in q_records]

            for m_start in range(0, num_mutants, self.mutant_chunk_size):
                m_end = min(num_mutants, m_start + self.mutant_chunk_size)
                chunk_mutants = list(range(m_start, m_end))
                flat_prompts: list[str] = []
                flat_mutant_indices: list[int] = []
                for mutant_idx in chunk_mutants:
                    flat_prompts.extend(q_prompts)
                    flat_mutant_indices.extend([mutant_idx] * len(q_prompts))

                mutant_tensor = torch.tensor(flat_mutant_indices, device=self.model.device, dtype=torch.long)
                texts, token_count = greedy_generate_texts(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompts=flat_prompts,
                    mutant_indices=mutant_tensor,
                    max_new_tokens=self.max_new_tokens,
                )
                generated_lengths = [len(self.tokenizer.encode(text, add_special_tokens=False)) for text in texts]
                profiler.record(
                    generated_tokens=token_count,
                    requests=len(flat_prompts),
                    mutant_evals=len(chunk_mutants),
                )

                for local_row, text in enumerate(texts):
                    local_mutant = local_row // len(q_prompts)
                    question_idx = local_row % len(q_prompts)
                    global_mutant = chunk_mutants[local_mutant]
                    result = score_record_prediction(
                        text,
                        q_records[question_idx],
                        reward_config=self.reward_config,
                    )
                    rewards[global_mutant] += result.reward
                    counts[global_mutant] += 1
                    exact_matches[global_mutant] += float(result.correct)
                    if collect_predictions:
                        predictions.append(
                            {
                                "id": q_ids[question_idx],
                                "question": q_questions[question_idx],
                                "gold_value": result.gold_value,
                                "prediction": text,
                                "reward": result.reward,
                                "predicted_value": result.predicted_value,
                                "correct": result.correct,
                                "generated_tokens": generated_lengths[local_row],
                                "mutant_index": global_mutant,
                            }
                        )

        counts = counts.clamp_min(1.0)
        return BatchExecutionResult(
            rewards=rewards / counts,
            exact_match_rates=exact_matches / counts,
            predictions=predictions,
            profiler_snapshot=profiler.snapshot(),
        )
