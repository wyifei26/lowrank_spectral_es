from __future__ import annotations

import time


class ThroughputProfiler:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.started_at = time.time()
        self.generated_tokens = 0
        self.requests = 0
        self.mutant_evals = 0

    def record(self, *, generated_tokens: int, requests: int, mutant_evals: int) -> None:
        self.generated_tokens += generated_tokens
        self.requests += requests
        self.mutant_evals += mutant_evals

    def snapshot(self) -> dict[str, float]:
        elapsed = max(time.time() - self.started_at, 1e-6)
        return {
            "elapsed_seconds": elapsed,
            "generated_tokens_total": float(self.generated_tokens),
            "requests_total": float(self.requests),
            "mutant_evals_total": float(self.mutant_evals),
            "tokens_per_sec": self.generated_tokens / elapsed,
            "requests_per_sec": self.requests / elapsed,
            "mutants_per_sec": self.mutant_evals / elapsed,
        }
