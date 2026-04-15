from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config_utils import load_yaml_config
from models.base_loader import resolve_dtype
from analyze_rl_projection import TensorStore, infer_svd_cache_path, infer_target_layer_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project an RL checkpoint into the base model spectral space and materialize W0 + delta W*."
    )
    parser.add_argument("--config", required=True, help="Resolved spectral-ES config path.")
    parser.add_argument("--rl-model", required=True, help="HF model directory for the RL checkpoint.")
    parser.add_argument("--output-dir", required=True, help="Directory to save the projected HF model.")
    parser.add_argument("--svd-cache", default=None, help="Optional explicit SVD cache path.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for projection math, e.g. cpu or cuda:0.",
    )
    parser.add_argument(
        "--save-dtype",
        default=None,
        help="Optional save dtype override. Defaults to config.model.dtype.",
    )
    return parser.parse_args()


def materialize_projected_model(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml_config(args.config)
    base_model_path = Path(config["model"]["model_path"]).resolve()
    rl_model_path = Path(args.rl_model).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    target_layer_names = infer_target_layer_names(config)
    svd_cache_path = Path(args.svd_cache).resolve() if args.svd_cache else infer_svd_cache_path(config, target_layer_names)
    svd_payload = torch.load(svd_cache_path, map_location="cpu")
    svd_layers: dict[str, dict[str, torch.Tensor]] = svd_payload["layers"]

    device = torch.device(args.device)
    save_dtype = resolve_dtype(args.save_dtype or config["model"].get("dtype", "bfloat16"))

    base_store = TensorStore(base_model_path)
    rl_store = TensorStore(rl_model_path)

    print(f"LOADING_BASE_MODEL path={base_model_path} dtype={save_dtype}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        trust_remote_code=True,
        torch_dtype=save_dtype,
        low_cpu_mem_usage=True,
    )
    model.to(torch.device("cpu"))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(base_model_path), trust_remote_code=True)

    projection_stats: list[dict[str, Any]] = []
    total_delta_norm_sq = 0.0
    total_residual_norm_sq = 0.0
    total_selected_weight_norm_sq = 0.0

    with torch.no_grad():
        with torch.inference_mode():
            with torch.cuda.device(device) if device.type == "cuda" else _nullcontext():
                from contextlib import ExitStack

                with ExitStack() as stack:
                    base_handles = base_store.open_handles(stack)
                    rl_handles = rl_store.open_handles(stack)
                    for layer_name in target_layer_names:
                        weight_name = f"{layer_name}.weight"
                        base_weight = base_store.get_tensor(base_handles, weight_name).to(device=device, dtype=torch.float32)
                        rl_weight = rl_store.get_tensor(rl_handles, weight_name).to(device=device, dtype=torch.float32)
                        delta_weight = rl_weight - base_weight

                        basis_bundle = svd_layers[layer_name]
                        u_basis = basis_bundle["u"].to(device=device, dtype=torch.float32)
                        vh_basis = basis_bundle["vh"].to(device=device, dtype=torch.float32)
                        projected_m = u_basis.transpose(0, 1).matmul(delta_weight).matmul(vh_basis.transpose(0, 1))
                        projected_delta = u_basis.matmul(projected_m).matmul(vh_basis)
                        residual = delta_weight - projected_delta

                        module = model.get_submodule(layer_name)
                        if module.weight.shape != projected_delta.shape:
                            raise ValueError(
                                f"shape mismatch for {layer_name}: model has {tuple(module.weight.shape)} "
                                f"but projected delta has {tuple(projected_delta.shape)}"
                            )
                        module.weight.add_(projected_delta.to(device="cpu", dtype=module.weight.dtype))

                        delta_norm = float(torch.linalg.vector_norm(delta_weight).item())
                        projected_norm = float(torch.linalg.vector_norm(projected_delta).item())
                        residual_norm = float(torch.linalg.vector_norm(residual).item())
                        selected_weight_norm = float(torch.linalg.vector_norm(base_weight).item())
                        total_delta_norm_sq += delta_norm * delta_norm
                        total_residual_norm_sq += residual_norm * residual_norm
                        total_selected_weight_norm_sq += selected_weight_norm * selected_weight_norm
                        projection_stats.append(
                            {
                                "layer_name": layer_name,
                                "delta_norm": delta_norm,
                                "projected_delta_norm": projected_norm,
                                "residual_norm": residual_norm,
                                "capture_ratio": projected_norm / delta_norm if delta_norm > 0 else None,
                                "residual_ratio": residual_norm / delta_norm if delta_norm > 0 else None,
                            }
                        )
                        print(
                            f"PROJECTED_LAYER name={layer_name} delta_norm={delta_norm:.6f} "
                            f"projected_norm={projected_norm:.6f} residual_ratio={residual_norm / delta_norm if delta_norm > 0 else float('nan'):.6e}",
                            flush=True,
                        )
                        del base_weight, rl_weight, delta_weight, u_basis, vh_basis, projected_m, projected_delta, residual

    print(f"SAVING_MODEL output_dir={output_dir}", flush=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    summary = {
        "config_path": str(Path(args.config).resolve()),
        "base_model_path": str(base_model_path),
        "rl_model_path": str(rl_model_path),
        "output_dir": str(output_dir),
        "svd_cache_path": str(svd_cache_path.resolve()),
        "device": str(device),
        "save_dtype": str(save_dtype).replace("torch.", ""),
        "num_projected_layers": len(projection_stats),
        "selected_weight_norm": total_selected_weight_norm_sq ** 0.5,
        "total_delta_norm": total_delta_norm_sq ** 0.5,
        "total_residual_norm": total_residual_norm_sq ** 0.5,
        "total_residual_ratio": (total_residual_norm_sq ** 0.5) / (total_delta_norm_sq ** 0.5)
        if total_delta_norm_sq > 0
        else None,
        "layers": projection_stats,
    }
    summary_path = output_dir / "projection_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    print(f"PROJECTION_SUMMARY {json.dumps(summary, ensure_ascii=True)}", flush=True)
    return summary


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def main() -> None:
    materialize_projected_model(parse_args())


if __name__ == "__main__":
    main()
