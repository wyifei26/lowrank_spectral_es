import json
from pathlib import Path

from engine.distributed_vllm_trainer import DistributedVLLMSpectralESTrainer


def test_write_train_predictions_keeps_only_preview_rows(tmp_path: Path):
    trainer = DistributedVLLMSpectralESTrainer.__new__(DistributedVLLMSpectralESTrainer)
    trainer.train_sample_root = tmp_path / "train_samples"
    trainer.rank = 0

    predictions = [{"id": index} for index in range(20)]
    output_path = trainer._write_train_predictions(step=3, predictions=predictions)

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert output_path == tmp_path / "train_samples" / "step_0003" / "rank_00.jsonl"
    assert len(rows) == 16
    assert rows == predictions[:16]


def test_resolve_eval_output_dir_groups_splits_under_single_root(tmp_path: Path):
    trainer = DistributedVLLMSpectralESTrainer.__new__(DistributedVLLMSpectralESTrainer)
    trainer.eval_output_root = tmp_path / "eval_outputs"

    val_path = trainer._resolve_eval_output_dir(split="val", step=0)
    test_path = trainer._resolve_eval_output_dir(split="test", step=12)

    assert val_path == tmp_path / "eval_outputs" / "val" / "step_0000"
    assert test_path == tmp_path / "eval_outputs" / "test" / "step_0012"
