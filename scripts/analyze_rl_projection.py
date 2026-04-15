from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw
from safetensors import safe_open
from scipy import stats
import torch
import torch.nn.functional as F

from config_utils import load_yaml_config


MODULE_ATTR_PATHS = {
    "q_proj": "self_attn.q_proj",
    "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
    "o_proj": "self_attn.o_proj",
    "gate_proj": "mlp.gate_proj",
    "up_proj": "mlp.up_proj",
    "down_proj": "mlp.down_proj",
}

ATTR_TO_MODULE_KEY = {value: key for key, value in MODULE_ATTR_PATHS.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project an RL checkpoint into the spectral-ES subspace and compare against ES M states."
    )
    parser.add_argument("--config", required=True, help="Resolved spectral-ES training config path.")
    parser.add_argument("--rl-model", required=True, help="HF model directory for the RL checkpoint.")
    parser.add_argument(
        "--es-checkpoint",
        action="append",
        required=True,
        help="Spectral-ES checkpoint path. Pass multiple times to compare multiple ES checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for JSON/CSV/Markdown outputs.",
    )
    parser.add_argument(
        "--svd-cache",
        default=None,
        help="Optional explicit SVD cache path. If omitted, infer from config and target layers.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Bootstrap samples for layer-level confidence intervals.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for projection math, e.g. cpu or cuda:0.",
    )
    parser.add_argument(
        "--heatmap-size",
        type=int,
        default=128,
        help="Adaptive pooled size for saved heatmaps and evolution strips.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip PNG heatmaps and trajectory plots.",
    )
    parser.add_argument(
        "--skip-correlation-stats",
        action="store_true",
        help="Skip per-layer/global Pearson and Spearman statistics for faster runs.",
    )
    return parser.parse_args()


def infer_target_layer_names(config: dict[str, Any]) -> list[str]:
    target_blocks = list(config["layers"]["target_blocks"])
    target_modules = list(config["layers"]["target_modules"])
    return [
        f"model.layers.{block_index}.{MODULE_ATTR_PATHS[module_key]}"
        for block_index in target_blocks
        for module_key in target_modules
    ]


def effective_requested_rank(config: dict[str, Any]) -> int:
    parameterization = str(config["subspace"].get("parameterization", "spectral_dense"))
    if parameterization == "full_factorized_m":
        return 0
    return int(config["subspace"]["rank"])


def infer_svd_cache_path(config: dict[str, Any], layer_names: list[str]) -> Path:
    rank = effective_requested_rank(config)
    band_strategy = str(config["subspace"]["band_strategy"])
    model_path = str(config["model"]["model_path"])
    cache_dir = Path(config["subspace"]["cache_dir"])
    rank_tag = "full" if rank == 0 else str(rank)
    signature = "|".join([model_path, rank_tag, band_strategy, *layer_names])
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"svd_cache_r{rank_tag}_{band_strategy}_{digest}.pt"


class TensorStore:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.single_file = self.model_dir / "model.safetensors"
        self.index_file = self.model_dir / "model.safetensors.index.json"
        self.weight_map: dict[str, str] | None = None

        if self.single_file.is_file():
            self.files = [self.single_file.name]
        elif self.index_file.is_file():
            with self.index_file.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            weight_map = payload.get("weight_map")
            if not isinstance(weight_map, dict):
                raise ValueError(f"invalid safetensors index: {self.index_file}")
            self.weight_map = {str(key): str(value) for key, value in weight_map.items()}
            self.files = sorted(set(self.weight_map.values()))
        else:
            raise FileNotFoundError(f"could not find model.safetensors or model.safetensors.index.json under {model_dir}")

    def open_handles(self, stack: ExitStack) -> dict[str, Any]:
        return {
            file_name: stack.enter_context(safe_open(str(self.model_dir / file_name), framework="pt", device="cpu"))
            for file_name in self.files
        }

    def get_tensor(self, handles: dict[str, Any], tensor_name: str) -> torch.Tensor:
        if self.weight_map is None:
            file_name = self.single_file.name
        else:
            file_name = self.weight_map.get(tensor_name)
            if file_name is None:
                raise KeyError(f"{tensor_name} is not present in {self.index_file}")
        return handles[file_name].get_tensor(tensor_name)


def build_rl_projection_cache(
    *,
    target_layer_names: list[str],
    svd_layers: dict[str, dict[str, torch.Tensor]],
    base_store: TensorStore,
    base_handles: dict[str, Any],
    rl_store: TensorStore,
    rl_handles: dict[str, Any],
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for layer_name in target_layer_names:
        if layer_name not in svd_layers:
            raise KeyError(f"{layer_name} missing from SVD cache")
        weight_name = f"{layer_name}.weight"
        base_weight = base_store.get_tensor(base_handles, weight_name).to(device=device, dtype=torch.float32)
        rl_weight = rl_store.get_tensor(rl_handles, weight_name).to(device=device, dtype=torch.float32)
        delta_weight = rl_weight - base_weight

        bundle = svd_layers[layer_name]
        u_basis = bundle["u"].to(device=device, dtype=torch.float32)
        vh_basis = bundle["vh"].to(device=device, dtype=torch.float32)
        v_basis = vh_basis.transpose(0, 1).contiguous()
        projected_m = u_basis.transpose(0, 1).matmul(delta_weight).matmul(v_basis)
        reconstructed_delta = u_basis.matmul(projected_m).matmul(vh_basis)
        residual_delta = delta_weight - reconstructed_delta

        full_delta_norm = float(torch.linalg.vector_norm(delta_weight).item())
        rl_proj_norm = float(torch.linalg.vector_norm(projected_m).item())
        residual_norm = float(torch.linalg.vector_norm(residual_delta).item())
        rl_diag_ratio = matrix_diag_energy_ratio(projected_m)
        rl_sym_ratio, rl_skew_ratio = matrix_symmetry_ratio(projected_m)

        cache[layer_name] = {
            "projected_m": projected_m.detach().cpu(),
            "rl_projected_norm": rl_proj_norm,
            "rl_full_delta_norm": full_delta_norm,
            "rl_residual_norm": residual_norm,
            "rl_projection_capture_ratio": safe_scalar_divide(rl_proj_norm, full_delta_norm),
            "rl_projection_residual_ratio": safe_scalar_divide(residual_norm, full_delta_norm),
            "rl_diag_energy_ratio": rl_diag_ratio,
            "rl_symmetric_ratio": rl_sym_ratio,
            "rl_skew_ratio": rl_skew_ratio,
            "rl_projected_stable_rank": matrix_stable_rank(projected_m),
        }
        del base_weight, rl_weight, delta_weight, u_basis, vh_basis, v_basis
        del projected_m, reconstructed_delta, residual_delta
    return cache


def safe_scalar_divide(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return math.nan
    return numerator / denominator


def flatten_cosine(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    lhs_flat = lhs.reshape(-1).float()
    rhs_flat = rhs.reshape(-1).float()
    lhs_norm = torch.linalg.vector_norm(lhs_flat).item()
    rhs_norm = torch.linalg.vector_norm(rhs_flat).item()
    if lhs_norm == 0.0 or rhs_norm == 0.0:
        return math.nan
    return float(torch.dot(lhs_flat, rhs_flat).item() / (lhs_norm * rhs_norm))


def safe_pearson(lhs: np.ndarray, rhs: np.ndarray) -> float:
    if lhs.size == 0 or rhs.size == 0:
        return math.nan
    if np.allclose(lhs, lhs[0]) or np.allclose(rhs, rhs[0]):
        return math.nan
    return float(np.corrcoef(lhs, rhs)[0, 1])


def safe_spearman(lhs: np.ndarray, rhs: np.ndarray) -> float:
    if lhs.size == 0 or rhs.size == 0:
        return math.nan
    if np.allclose(lhs, lhs[0]) or np.allclose(rhs, rhs[0]):
        return math.nan
    return float(stats.spearmanr(lhs, rhs).statistic)


def matrix_diag_energy_ratio(matrix: torch.Tensor) -> float:
    total = float(torch.sum(matrix.float().pow(2)).item())
    if total == 0.0:
        return math.nan
    diag = float(torch.sum(torch.diagonal(matrix.float()).pow(2)).item())
    return diag / total


def matrix_symmetry_ratio(matrix: torch.Tensor) -> tuple[float, float]:
    matrix = matrix.float()
    total_norm = float(torch.linalg.vector_norm(matrix).item())
    if total_norm == 0.0:
        return math.nan, math.nan
    symmetric = 0.5 * (matrix + matrix.transpose(0, 1))
    skew = 0.5 * (matrix - matrix.transpose(0, 1))
    return (
        float(torch.linalg.vector_norm(symmetric).item() / total_norm),
        float(torch.linalg.vector_norm(skew).item() / total_norm),
    )


def matrix_stable_rank(matrix: torch.Tensor) -> float:
    singular_values = torch.linalg.svdvals(matrix.float())
    if singular_values.numel() == 0:
        return math.nan
    spectral_norm = float(singular_values.max().item())
    if spectral_norm == 0.0:
        return math.nan
    fro_sq = float(torch.sum(singular_values.pow(2)).item())
    return fro_sq / (spectral_norm * spectral_norm)


def bootstrap_ci(values: np.ndarray, *, bootstrap_samples: int, seed: int) -> tuple[float, float]:
    if values.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(bootstrap_samples):
        sample = values[rng.integers(0, values.size, size=values.size)]
        samples.append(float(np.mean(sample)))
    lower, upper = np.quantile(np.asarray(samples, dtype=np.float64), [0.025, 0.975])
    return float(lower), float(upper)


def layer_level_significance(values: np.ndarray) -> dict[str, float]:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return {
            "t_statistic": math.nan,
            "t_pvalue_greater_than_zero": math.nan,
            "wilcoxon_statistic": math.nan,
            "wilcoxon_pvalue_greater_than_zero": math.nan,
            "positive_rate": math.nan,
            "positive_sign_test_pvalue": math.nan,
        }

    positive_count = int(np.sum(finite_values > 0.0))
    positive_rate = positive_count / finite_values.size
    t_stat = stats.ttest_1samp(finite_values, popmean=0.0, alternative="greater")

    try:
        wilcoxon = stats.wilcoxon(finite_values, alternative="greater", zero_method="zsplit")
        wilcoxon_statistic = float(wilcoxon.statistic)
        wilcoxon_pvalue = float(wilcoxon.pvalue)
    except ValueError:
        wilcoxon_statistic = math.nan
        wilcoxon_pvalue = math.nan

    sign_test = stats.binomtest(positive_count, finite_values.size, p=0.5, alternative="greater")
    return {
        "t_statistic": float(t_stat.statistic),
        "t_pvalue_greater_than_zero": float(t_stat.pvalue),
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_pvalue_greater_than_zero": wilcoxon_pvalue,
        "positive_rate": float(positive_rate),
        "positive_sign_test_pvalue": float(sign_test.pvalue),
    }


def optimal_scale_metrics(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[float, float]:
    lhs_flat = lhs.reshape(-1).float()
    rhs_flat = rhs.reshape(-1).float()
    lhs_sq = float(torch.dot(lhs_flat, lhs_flat).item())
    rhs_norm = float(torch.linalg.vector_norm(rhs_flat).item())
    if lhs_sq == 0.0 or rhs_norm == 0.0:
        return math.nan, math.nan
    alpha = float(torch.dot(lhs_flat, rhs_flat).item() / lhs_sq)
    residual = rhs_flat - alpha * lhs_flat
    residual_ratio = float(torch.linalg.vector_norm(residual).item() / rhs_norm)
    return alpha, residual_ratio


def optimal_scale_metrics_from_accumulators(dot_value: float, lhs_sq: float, rhs_sq: float) -> tuple[float, float]:
    if lhs_sq == 0.0 or rhs_sq == 0.0:
        return math.nan, math.nan
    alpha = dot_value / lhs_sq
    residual_sq = max(rhs_sq - 2.0 * alpha * dot_value + (alpha * alpha * lhs_sq), 0.0)
    residual_ratio = math.sqrt(residual_sq) / math.sqrt(rhs_sq)
    return float(alpha), float(residual_ratio)


def adaptive_downsample(matrix: torch.Tensor, target_size: int) -> torch.Tensor:
    matrix = matrix.float()
    height, width = matrix.shape
    target_h = max(1, min(int(target_size), height))
    target_w = max(1, min(int(target_size), width))
    pooled = F.adaptive_avg_pool2d(matrix.unsqueeze(0).unsqueeze(0), (target_h, target_w))
    return pooled[0, 0].cpu()


def symmetric_limit(matrices: list[torch.Tensor], *, quantile: float = 0.995) -> float:
    values = []
    for matrix in matrices:
        flat = matrix.reshape(-1).abs().cpu().numpy()
        if flat.size:
            values.append(np.quantile(flat, quantile))
    if not values:
        return 1.0
    limit = float(max(values))
    return limit if limit > 0.0 else 1.0


def matrix_to_heatmap_image(matrix: torch.Tensor, *, limit: float, pixel_size: int = 512) -> Image.Image:
    matrix = matrix.float().cpu()
    clipped = torch.clamp(matrix / limit, min=-1.0, max=1.0).numpy()
    rgb = np.zeros((*clipped.shape, 3), dtype=np.uint8)

    negative = clipped < 0
    positive = clipped > 0
    neutral = ~(negative | positive)

    neg_strength = np.clip(-clipped[negative], 0.0, 1.0)
    pos_strength = np.clip(clipped[positive], 0.0, 1.0)

    rgb[neutral] = np.array([255, 255, 255], dtype=np.uint8)
    rgb[negative, 0] = (255 * (1.0 - neg_strength)).astype(np.uint8)
    rgb[negative, 1] = (255 * (1.0 - neg_strength)).astype(np.uint8)
    rgb[negative, 2] = 255
    rgb[positive, 0] = 255
    rgb[positive, 1] = (255 * (1.0 - pos_strength)).astype(np.uint8)
    rgb[positive, 2] = (255 * (1.0 - pos_strength)).astype(np.uint8)

    image = Image.fromarray(rgb, mode="RGB")
    if image.size != (pixel_size, pixel_size):
        image = image.resize((pixel_size, pixel_size), resample=Image.Resampling.NEAREST)
    return image


def add_title(image: Image.Image, title: str) -> Image.Image:
    title_height = 26
    canvas = Image.new("RGB", (image.width, image.height + title_height), color=(255, 255, 255))
    canvas.paste(image, (0, title_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill=(0, 0, 0))
    return canvas


def save_strip_image(images: list[Image.Image], path: Path) -> None:
    if not images:
        return
    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    offset = 0
    for image in images:
        canvas.paste(image, (offset, 0))
        offset += image.width
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_series_plot(
    *,
    x_values: list[int],
    series: list[tuple[str, list[float], tuple[int, int, int]]],
    title: str,
    y_label: str,
    path: Path,
) -> None:
    width = 1200
    height = 720
    margin_left = 90
    margin_right = 24
    margin_top = 48
    margin_bottom = 72
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    all_points = [
        value
        for _, values, _ in series
        for value in values
        if value is not None and math.isfinite(value)
    ]
    if not x_values or not all_points:
        draw.text((20, 20), f"{title}: no finite data", fill=(0, 0, 0))
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        return

    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(all_points)
    max_y = max(all_points)
    if min_x == max_x:
        max_x = min_x + 1
    if min_y == max_y:
        pad = 1.0 if min_y == 0.0 else abs(min_y) * 0.1
        min_y -= pad
        max_y += pad

    plot_left = margin_left
    plot_top = margin_top
    plot_right = width - margin_right
    plot_bottom = height - margin_bottom

    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=(0, 0, 0), width=2)
    draw.text((plot_left, 14), title, fill=(0, 0, 0))
    draw.text((20, (plot_top + plot_bottom) // 2), y_label, fill=(0, 0, 0))
    draw.text(((plot_left + plot_right) // 2, height - 28), "step", fill=(0, 0, 0))

    def project(step: int, value: float) -> tuple[int, int]:
        x_ratio = (step - min_x) / (max_x - min_x)
        y_ratio = (value - min_y) / (max_y - min_y)
        px = int(plot_left + x_ratio * (plot_right - plot_left))
        py = int(plot_bottom - y_ratio * (plot_bottom - plot_top))
        return px, py

    for tick_ratio in np.linspace(0.0, 1.0, num=5):
        y_value = min_y + tick_ratio * (max_y - min_y)
        _, py = project(min_x, y_value)
        draw.line((plot_left, py, plot_right, py), fill=(230, 230, 230), width=1)
        draw.text((12, py - 8), f"{y_value:.3f}", fill=(0, 0, 0))

    for step in x_values:
        px, _ = project(step, min_y)
        draw.line((px, plot_top, px, plot_bottom), fill=(245, 245, 245), width=1)
        draw.text((px - 10, plot_bottom + 8), str(step), fill=(0, 0, 0))

    legend_x = plot_left
    legend_y = plot_bottom + 30
    for label, values, color in series:
        points = [
            project(step, value)
            for step, value in zip(x_values, values)
            if value is not None and math.isfinite(value)
        ]
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        for point in points:
            draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=color)
        draw.rectangle((legend_x, legend_y + 5, legend_x + 18, legend_y + 18), fill=color)
        draw.text((legend_x + 24, legend_y), label, fill=(0, 0, 0))
        legend_x += max(170, 16 + 8 * len(label))


def parse_layer_identity(layer_name: str) -> tuple[int, str, str]:
    parts = layer_name.split(".")
    if len(parts) < 5:
        raise ValueError(f"unexpected layer name: {layer_name}")
    block_index = int(parts[2])
    attr_path = ".".join(parts[3:])
    module_key = ATTR_TO_MODULE_KEY.get(attr_path, attr_path)
    return block_index, module_key, attr_path


def analyze_single_checkpoint(
    *,
    checkpoint_path: Path,
    adapter_state: dict[str, dict[str, torch.Tensor]],
    target_layer_names: list[str],
    rl_projection_cache: dict[str, dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
    device: torch.device,
    compute_correlation_stats: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, torch.Tensor]]]:
    layer_rows: list[dict[str, Any]] = []
    module_rows_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    module_matrix_sums: dict[str, dict[str, torch.Tensor | int]] = {}

    es_vectors: list[np.ndarray] = []
    rl_vectors: list[np.ndarray] = []
    layer_cosines: list[float] = []
    global_dot_sum = 0.0
    global_es_sq_sum = 0.0
    global_rl_sq_sum = 0.0
    es_sq_sum = 0.0

    for layer_name in target_layer_names:
        if layer_name not in adapter_state:
            raise KeyError(f"{layer_name} missing from {checkpoint_path}")
        if layer_name not in rl_projection_cache:
            raise KeyError(f"{layer_name} missing from RL projection cache")
        cache_entry = rl_projection_cache[layer_name]
        projected_m_cpu = cache_entry["projected_m"]
        projected_m = projected_m_cpu.to(device=device, dtype=torch.float32)

        state_payload = adapter_state[layer_name]
        if "effective_m_state" in state_payload:
            es_m = state_payload["effective_m_state"].float()
        elif "m_state" in state_payload and state_payload["m_state"].ndim == 2:
            es_m = state_payload["m_state"].float()
        else:
            raise ValueError(
                f"{layer_name} in {checkpoint_path} does not expose a comparable spectral M matrix; "
                "this analysis currently supports dense spectral states or factorized spectral states only"
            )
        es_m_device = es_m.to(device=device, dtype=torch.float32)

        block_index, module_key, attr_path = parse_layer_identity(layer_name)
        full_delta_norm = float(cache_entry["rl_full_delta_norm"])
        rl_proj_norm = float(cache_entry["rl_projected_norm"])
        residual_norm = float(cache_entry["rl_residual_norm"])
        es_norm = float(torch.linalg.vector_norm(es_m_device).item())
        cosine = flatten_cosine(es_m_device, projected_m)
        dot_value = float(torch.dot(es_m_device.reshape(-1), projected_m.reshape(-1)).item())
        global_dot_sum += dot_value
        global_es_sq_sum += es_norm * es_norm
        global_rl_sq_sum += rl_proj_norm * rl_proj_norm

        if compute_correlation_stats:
            es_vector = es_m.reshape(-1).cpu().numpy()
            rl_vector = projected_m_cpu.reshape(-1).numpy()
            layer_pearson = safe_pearson(es_vector, rl_vector)
            layer_spearman = safe_spearman(es_vector, rl_vector)
            es_vectors.append(es_vector)
            rl_vectors.append(rl_vector)
        else:
            layer_pearson = math.nan
            layer_spearman = math.nan

        es_diag_ratio = matrix_diag_energy_ratio(es_m_device)
        rl_diag_ratio = matrix_diag_energy_ratio(projected_m)
        es_sym_ratio, es_skew_ratio = matrix_symmetry_ratio(es_m_device)
        rl_sym_ratio, rl_skew_ratio = matrix_symmetry_ratio(projected_m)

        row = {
            "layer_name": layer_name,
            "block_index": block_index,
            "module_key": module_key,
            "attr_path": attr_path,
            "es_norm": es_norm,
            "rl_projected_norm": rl_proj_norm,
            "rl_full_delta_norm": full_delta_norm,
            "rl_residual_norm": residual_norm,
            "rl_projection_capture_ratio": float(cache_entry["rl_projection_capture_ratio"]),
            "rl_projection_residual_ratio": float(cache_entry["rl_projection_residual_ratio"]),
            "es_vs_rl_cosine": cosine,
            "es_vs_rl_pearson": layer_pearson,
            "es_vs_rl_spearman": layer_spearman,
            "es_over_rl_projected_norm_ratio": safe_scalar_divide(es_norm, rl_proj_norm),
            "m_difference_norm": float(torch.linalg.vector_norm(es_m_device - projected_m).item()),
            "es_diag_energy_ratio": es_diag_ratio,
            "rl_diag_energy_ratio": float(cache_entry["rl_diag_energy_ratio"]),
            "diag_ratio_gap": abs(es_diag_ratio - rl_diag_ratio) if not (math.isnan(es_diag_ratio) or math.isnan(rl_diag_ratio)) else math.nan,
            "es_symmetric_ratio": es_sym_ratio,
            "es_skew_ratio": es_skew_ratio,
            "rl_symmetric_ratio": float(cache_entry["rl_symmetric_ratio"]),
            "rl_skew_ratio": float(cache_entry["rl_skew_ratio"]),
            "es_stable_rank": matrix_stable_rank(es_m_device),
            "rl_projected_stable_rank": float(cache_entry["rl_projected_stable_rank"]),
        }
        layer_rows.append(row)
        module_rows_map[module_key].append(row)
        if module_key not in module_matrix_sums:
            module_matrix_sums[module_key] = {
                "es_sum": torch.zeros_like(es_m, dtype=torch.float32),
                "rl_sum": torch.zeros_like(projected_m_cpu, dtype=torch.float32),
                "count": 0,
            }
        module_matrix_sums[module_key]["es_sum"] += es_m
        module_matrix_sums[module_key]["rl_sum"] += projected_m_cpu
        module_matrix_sums[module_key]["count"] += 1

        layer_cosines.append(cosine)
        es_sq_sum += es_norm * es_norm
        del projected_m, es_m_device

    layer_cos_array = np.asarray(layer_cosines, dtype=np.float64)
    layer_cos_finite = layer_cos_array[np.isfinite(layer_cos_array)]
    cosine_ci_low, cosine_ci_high = bootstrap_ci(
        layer_cos_finite,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    significance = layer_level_significance(layer_cos_array)
    optimal_alpha, optimal_scaled_residual_ratio = optimal_scale_metrics_from_accumulators(
        global_dot_sum,
        global_es_sq_sum,
        global_rl_sq_sum,
    )
    global_cosine = safe_scalar_divide(global_dot_sum, math.sqrt(global_es_sq_sum * global_rl_sq_sum))
    if compute_correlation_stats and es_vectors and rl_vectors:
        global_es = np.concatenate(es_vectors)
        global_rl = np.concatenate(rl_vectors)
        global_pearson = safe_pearson(global_es, global_rl)
        global_spearman = safe_spearman(global_es, global_rl)
    else:
        global_pearson = math.nan
        global_spearman = math.nan

    module_rows: list[dict[str, Any]] = []
    module_matrices: dict[str, dict[str, torch.Tensor]] = {}
    for module_key, rows in sorted(module_rows_map.items()):
        cosine_values = np.asarray([row["es_vs_rl_cosine"] for row in rows], dtype=np.float64)
        capture_values = np.asarray([row["rl_projection_capture_ratio"] for row in rows], dtype=np.float64)
        es_norm_values = np.asarray([row["es_norm"] for row in rows], dtype=np.float64)
        rl_norm_values = np.asarray([row["rl_projected_norm"] for row in rows], dtype=np.float64)
        full_delta_values = np.asarray([row["rl_full_delta_norm"] for row in rows], dtype=np.float64)
        diag_gap_values = np.asarray([row["diag_ratio_gap"] for row in rows], dtype=np.float64)
        packed = module_matrix_sums[module_key]
        count = int(packed["count"])
        es_mean = packed["es_sum"] / count
        rl_mean = packed["rl_sum"] / count
        diff_mean = es_mean - rl_mean
        module_alpha, module_scaled_residual_ratio = optimal_scale_metrics(es_mean, rl_mean)
        module_matrices[module_key] = {
            "es_mean": es_mean.cpu(),
            "rl_mean": rl_mean.cpu(),
            "diff_mean": diff_mean.cpu(),
        }
        module_rows.append(
            {
                "module_key": module_key,
                "num_layers": len(rows),
                "mean_cosine": float(np.nanmean(cosine_values)),
                "median_cosine": float(np.nanmedian(cosine_values)),
                "positive_cosine_rate": float(np.nanmean(cosine_values > 0.0)),
                "mean_projection_capture_ratio": float(np.nanmean(capture_values)),
                "energy_projection_capture_ratio": safe_scalar_divide(
                    float(np.sum(rl_norm_values * rl_norm_values)),
                    float(np.sum(full_delta_values * full_delta_values)),
                ),
                "mean_es_norm": float(np.nanmean(es_norm_values)),
                "mean_rl_projected_norm": float(np.nanmean(rl_norm_values)),
                "mean_diag_ratio_gap": float(np.nanmean(diag_gap_values)),
                "module_mean_matrix_cosine": flatten_cosine(es_mean, rl_mean),
                "module_mean_matrix_optimal_alpha": module_alpha,
                "module_mean_matrix_scaled_residual_ratio": module_scaled_residual_ratio,
            }
        )

    ranked_rows = sorted(
        layer_rows,
        key=lambda item: (-math.inf if math.isnan(item["es_vs_rl_cosine"]) else item["es_vs_rl_cosine"]),
    )
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "num_layers": len(layer_rows),
        "global_cosine": global_cosine,
        "global_pearson": global_pearson,
        "global_spearman": global_spearman,
        "global_optimal_alpha": optimal_alpha,
        "global_scaled_residual_ratio": optimal_scaled_residual_ratio,
        "layer_mean_cosine": float(np.nanmean(layer_cos_array)),
        "layer_median_cosine": float(np.nanmedian(layer_cos_array)),
        "layer_mean_cosine_ci95": [cosine_ci_low, cosine_ci_high],
        "projection_energy_capture_ratio": safe_scalar_divide(global_rl_sq_sum, sum(float(item["rl_full_delta_norm"]) ** 2 for item in rl_projection_cache.values())),
        "residual_energy_ratio": safe_scalar_divide(
            sum(float(item["rl_residual_norm"]) ** 2 for item in rl_projection_cache.values()),
            sum(float(item["rl_full_delta_norm"]) ** 2 for item in rl_projection_cache.values()),
        ),
        "es_over_rl_projected_energy_ratio": safe_scalar_divide(es_sq_sum, global_rl_sq_sum),
        "layer_mean_projection_capture_ratio": float(
            np.nanmean(np.asarray([row["rl_projection_capture_ratio"] for row in layer_rows], dtype=np.float64))
        ),
        "layer_mean_diag_ratio_gap": float(
            np.nanmean(np.asarray([row["diag_ratio_gap"] for row in layer_rows], dtype=np.float64))
        ),
        "alignment_significance": significance,
        "top_aligned_layers": ranked_rows[-10:][::-1],
        "top_misaligned_layers": ranked_rows[:10],
    }
    return summary, layer_rows, module_rows, module_matrices


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_checkpoint_heatmaps(
    *,
    checkpoint_name: str,
    module_matrices: dict[str, dict[str, torch.Tensor]],
    output_dir: Path,
    heatmap_size: int,
) -> None:
    for module_key, matrices in sorted(module_matrices.items()):
        es_mean = adaptive_downsample(matrices["es_mean"], heatmap_size)
        rl_mean = adaptive_downsample(matrices["rl_mean"], heatmap_size)
        diff_mean = adaptive_downsample(matrices["diff_mean"], heatmap_size)
        limit = symmetric_limit([es_mean, rl_mean, diff_mean])
        es_image = add_title(matrix_to_heatmap_image(es_mean, limit=limit), f"{module_key} ES mean")
        rl_image = add_title(matrix_to_heatmap_image(rl_mean, limit=limit), f"{module_key} RL mean")
        diff_image = add_title(matrix_to_heatmap_image(diff_mean, limit=limit), f"{module_key} ES-RL")
        es_path = output_dir / f"{checkpoint_name}_{module_key}_es_mean_heatmap.png"
        rl_path = output_dir / f"{checkpoint_name}_{module_key}_rl_mean_heatmap.png"
        diff_path = output_dir / f"{checkpoint_name}_{module_key}_diff_mean_heatmap.png"
        es_path.parent.mkdir(parents=True, exist_ok=True)
        es_image.save(es_path)
        rl_image.save(rl_path)
        diff_image.save(diff_path)
        save_strip_image(
            [es_image, rl_image, diff_image],
            output_dir / f"{checkpoint_name}_{module_key}_comparison_strip.png",
        )


def save_evolution_heatmaps(
    *,
    module_history: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    heatmap_size: int,
) -> None:
    for module_key, records in sorted(module_history.items()):
        if not records:
            continue
        records = sorted(records, key=lambda item: int(item["step"]))
        rl_target = adaptive_downsample(records[0]["rl_mean"], heatmap_size)
        es_mats = [adaptive_downsample(record["es_mean"], heatmap_size) for record in records]
        diff_mats = [adaptive_downsample(record["diff_mean"], heatmap_size) for record in records]
        limit = symmetric_limit([rl_target, *es_mats, *diff_mats])

        es_images = [
            add_title(matrix_to_heatmap_image(matrix, limit=limit), f"step {record['step']}")
            for record, matrix in zip(records, es_mats)
        ]
        diff_images = [
            add_title(matrix_to_heatmap_image(matrix, limit=limit), f"step {record['step']}")
            for record, matrix in zip(records, diff_mats)
        ]
        rl_image = add_title(matrix_to_heatmap_image(rl_target, limit=limit), "RL target")
        save_strip_image(
            es_images + [rl_image],
            output_dir / f"{module_key}_es_evolution_strip.png",
        )
        save_strip_image(
            diff_images + [rl_image],
            output_dir / f"{module_key}_diff_evolution_strip.png",
        )


def trajectory_stat(steps: np.ndarray, values: np.ndarray) -> dict[str, float]:
    finite_mask = np.isfinite(values)
    finite_steps = steps[finite_mask]
    finite_values = values[finite_mask]
    if finite_values.size < 2:
        return {
            "start": math.nan,
            "end": math.nan,
            "delta": math.nan,
            "linear_slope_per_step": math.nan,
            "linear_pvalue": math.nan,
            "spearman_rho": math.nan,
            "spearman_pvalue": math.nan,
        }
    regression = stats.linregress(finite_steps, finite_values)
    spearman = stats.spearmanr(finite_steps, finite_values)
    return {
        "start": float(finite_values[0]),
        "end": float(finite_values[-1]),
        "delta": float(finite_values[-1] - finite_values[0]),
        "linear_slope_per_step": float(regression.slope),
        "linear_pvalue": float(regression.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_pvalue": float(spearman.pvalue),
    }


def build_trajectory_summary(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        return {}
    ordered = sorted(summaries, key=lambda item: int(item["step"]))
    steps = np.asarray([int(item["step"]) for item in ordered], dtype=np.float64)
    global_cosine = np.asarray([float(item["global_cosine"]) for item in ordered], dtype=np.float64)
    layer_mean_cosine = np.asarray([float(item["layer_mean_cosine"]) for item in ordered], dtype=np.float64)
    positive_rate = np.asarray(
        [float(item["alignment_significance"]["positive_rate"]) for item in ordered],
        dtype=np.float64,
    )
    scaled_residual = np.asarray([float(item["global_scaled_residual_ratio"]) for item in ordered], dtype=np.float64)
    return {
        "num_checkpoints": len(ordered),
        "steps": [int(step) for step in steps.tolist()],
        "global_cosine_trend": trajectory_stat(steps, global_cosine),
        "layer_mean_cosine_trend": trajectory_stat(steps, layer_mean_cosine),
        "positive_rate_trend": trajectory_stat(steps, positive_rate),
        "global_scaled_residual_trend": trajectory_stat(steps, scaled_residual),
    }


def render_report(
    *,
    config_path: Path,
    base_model_path: Path,
    rl_model_path: Path,
    svd_cache_path: Path,
    summaries: list[dict[str, Any]],
    trajectory_summary: dict[str, Any] | None,
    output_dir: Path,
) -> str:
    lines = [
        "# RL Projection vs Spectral-ES",
        "",
        f"- Config: `{config_path}`",
        f"- Base model: `{base_model_path}`",
        f"- RL model: `{rl_model_path}`",
        f"- SVD cache: `{svd_cache_path}`",
        "",
    ]
    if trajectory_summary:
        global_trend = trajectory_summary["global_cosine_trend"]
        layer_trend = trajectory_summary["layer_mean_cosine_trend"]
        positive_trend = trajectory_summary["positive_rate_trend"]
        residual_trend = trajectory_summary["global_scaled_residual_trend"]
        lines.extend(
            [
                "## Trajectory Summary",
                "",
                f"- Checked steps: `{trajectory_summary['steps']}`",
                f"- Global cosine delta: `{global_trend['delta']:.4f}` "
                f"(slope/step `{global_trend['linear_slope_per_step']:.4e}`, p=`{global_trend['linear_pvalue']:.3e}`)",
                f"- Layer mean cosine delta: `{layer_trend['delta']:.4f}` "
                f"(slope/step `{layer_trend['linear_slope_per_step']:.4e}`, p=`{layer_trend['linear_pvalue']:.3e}`)",
                f"- Positive layer cosine rate delta: `{positive_trend['delta']:.4f}`",
                f"- Optimal-scale residual delta: `{residual_trend['delta']:.4f}`",
                "",
            ]
        )
    for summary in summaries:
        checkpoint_name = Path(summary["checkpoint_path"]).stem
        significance = summary["alignment_significance"]
        lines.extend(
            [
                f"## {checkpoint_name}",
                "",
                f"- Global cosine: `{summary['global_cosine']:.4f}`",
                f"- Layer mean cosine: `{summary['layer_mean_cosine']:.4f}` "
                f"(95% CI `{summary['layer_mean_cosine_ci95'][0]:.4f}` to `{summary['layer_mean_cosine_ci95'][1]:.4f}`)",
                f"- Positive layer cosine rate: `{significance['positive_rate']:.4f}`",
                f"- Projection energy capture ratio: `{summary['projection_energy_capture_ratio']:.4f}`",
                f"- Residual energy ratio: `{summary['residual_energy_ratio']:.4f}`",
                f"- ES / RL projected energy ratio: `{summary['es_over_rl_projected_energy_ratio']:.4f}`",
                f"- Optimal global rescale alpha: `{summary['global_optimal_alpha']:.4e}`",
                f"- Residual after optimal rescale: `{summary['global_scaled_residual_ratio']:.4f}`",
                f"- Sign-test p-value (alignment > 0): `{significance['positive_sign_test_pvalue']:.3e}`",
                f"- T-test p-value (mean cosine > 0): `{significance['t_pvalue_greater_than_zero']:.3e}`",
                "",
                "Top aligned layers:",
            ]
        )
        for row in summary["top_aligned_layers"][:5]:
            lines.append(
                f"- `{row['layer_name']}` cosine=`{row['es_vs_rl_cosine']:.4f}` "
                f"capture=`{row['rl_projection_capture_ratio']:.4f}`"
            )
        lines.append("")
        lines.append("Top misaligned layers:")
        for row in summary["top_misaligned_layers"][:5]:
            lines.append(
                f"- `{row['layer_name']}` cosine=`{row['es_vs_rl_cosine']:.4f}` "
                f"capture=`{row['rl_projection_capture_ratio']:.4f}`"
            )
        lines.append("")
        lines.append(
            f"Artifacts: `{output_dir / f'{checkpoint_name}_summary.json'}`, "
            f"`{output_dir / f'{checkpoint_name}_layer_metrics.csv'}`, "
            f"`{output_dir / f'{checkpoint_name}_module_metrics.csv'}`"
        )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    rl_model_path = Path(args.rl_model).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device {args.device}, but CUDA is not available")

    config = load_yaml_config(config_path)
    base_model_path = Path(config["model"]["model_path"]).resolve()
    target_layer_names = infer_target_layer_names(config)
    svd_cache_path = Path(args.svd_cache).resolve() if args.svd_cache else infer_svd_cache_path(config, target_layer_names)

    checkpoint_payloads: list[tuple[Path, dict[str, Any]]] = []
    for raw_checkpoint_path in args.es_checkpoint:
        checkpoint_path = Path(raw_checkpoint_path).resolve()
        payload = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_payloads.append((checkpoint_path, payload))
    checkpoint_payloads.sort(key=lambda item: int(item[1]["step"]))

    adapter_layer_names = list(checkpoint_payloads[0][1]["adapter_state"].keys())
    if adapter_layer_names != target_layer_names:
        raise ValueError("adapter_state layer ordering does not match config-derived target layer ordering")

    svd_payload = torch.load(svd_cache_path, map_location="cpu")
    svd_layers = svd_payload["layers"]

    base_store = TensorStore(base_model_path)
    rl_store = TensorStore(rl_model_path)

    summaries: list[dict[str, Any]] = []
    module_trajectory_rows: list[dict[str, Any]] = []
    module_matrix_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with ExitStack() as stack:
        base_handles = base_store.open_handles(stack)
        rl_handles = rl_store.open_handles(stack)
        rl_projection_cache = build_rl_projection_cache(
            target_layer_names=target_layer_names,
            svd_layers=svd_layers,
            base_store=base_store,
            base_handles=base_handles,
            rl_store=rl_store,
            rl_handles=rl_handles,
            device=device,
        )

        for checkpoint_path, payload in checkpoint_payloads:
            summary, layer_rows, module_rows, module_matrices = analyze_single_checkpoint(
                checkpoint_path=checkpoint_path,
                adapter_state=payload["adapter_state"],
                target_layer_names=target_layer_names,
                rl_projection_cache=rl_projection_cache,
                bootstrap_samples=int(args.bootstrap_samples),
                seed=int(args.seed),
                device=device,
                compute_correlation_stats=not args.skip_correlation_stats,
            )
            checkpoint_name = checkpoint_path.stem
            summary["step"] = int(payload["step"])
            summary["best_val_accuracy"] = float(payload["best_val_accuracy"])
            summaries.append(summary)

            write_json(output_dir / f"{checkpoint_name}_summary.json", summary)
            write_csv(output_dir / f"{checkpoint_name}_layer_metrics.csv", layer_rows)
            write_csv(output_dir / f"{checkpoint_name}_module_metrics.csv", module_rows)
            for row in module_rows:
                enriched = dict(row)
                enriched["step"] = int(payload["step"])
                enriched["checkpoint_name"] = checkpoint_name
                module_trajectory_rows.append(enriched)
            for module_key, matrices in module_matrices.items():
                module_matrix_history[module_key].append(
                    {
                        "step": int(payload["step"]),
                        "checkpoint_name": checkpoint_name,
                        "es_mean": adaptive_downsample(matrices["es_mean"], int(args.heatmap_size)),
                        "rl_mean": adaptive_downsample(matrices["rl_mean"], int(args.heatmap_size)),
                        "diff_mean": adaptive_downsample(matrices["diff_mean"], int(args.heatmap_size)),
                    }
                )
            if not args.skip_plots:
                save_checkpoint_heatmaps(
                    checkpoint_name=checkpoint_name,
                    module_matrices=module_matrices,
                    output_dir=output_dir,
                    heatmap_size=int(args.heatmap_size),
                )

    trajectory_summary = build_trajectory_summary(summaries)
    if module_trajectory_rows:
        write_csv(output_dir / "trajectory_module_metrics.csv", module_trajectory_rows)
    write_json(output_dir / "trajectory_summary.json", trajectory_summary)
    if not args.skip_plots:
        save_evolution_heatmaps(
            module_history=module_matrix_history,
            output_dir=output_dir,
            heatmap_size=int(args.heatmap_size),
        )
        ordered_summaries = sorted(summaries, key=lambda item: int(item["step"]))
        steps = [int(summary["step"]) for summary in ordered_summaries]
        save_series_plot(
            x_values=steps,
            series=[
                ("global cosine", [float(summary["global_cosine"]) for summary in ordered_summaries], (30, 90, 200)),
                (
                    "layer mean cosine",
                    [float(summary["layer_mean_cosine"]) for summary in ordered_summaries],
                    (220, 80, 60),
                ),
            ],
            title="Alignment trajectory",
            y_label="cosine",
            path=output_dir / "alignment_trajectory.png",
        )
        save_series_plot(
            x_values=steps,
            series=[
                (
                    "positive layer rate",
                    [float(summary["alignment_significance"]["positive_rate"]) for summary in ordered_summaries],
                    (40, 150, 90),
                ),
                (
                    "scaled residual",
                    [float(summary["global_scaled_residual_ratio"]) for summary in ordered_summaries],
                    (130, 70, 170),
                ),
            ],
            title="Directionality diagnostics",
            y_label="metric",
            path=output_dir / "directionality_diagnostics.png",
        )
        module_keys = sorted({row["module_key"] for row in module_trajectory_rows})
        palette = [
            (30, 90, 200),
            (220, 80, 60),
            (40, 150, 90),
            (130, 70, 170),
            (220, 170, 0),
            (0, 150, 160),
            (140, 90, 40),
        ]
        module_series = []
        for idx, module_key in enumerate(module_keys):
            rows = sorted(
                [row for row in module_trajectory_rows if row["module_key"] == module_key],
                key=lambda item: int(item["step"]),
            )
            module_series.append(
                (
                    module_key,
                    [float(row["module_mean_matrix_cosine"]) for row in rows],
                    palette[idx % len(palette)],
                )
            )
        save_series_plot(
            x_values=steps,
            series=module_series,
            title="Module mean-matrix cosine trajectory",
            y_label="cosine",
            path=output_dir / "module_mean_cosine_trajectory.png",
        )

    report = render_report(
        config_path=config_path,
        base_model_path=base_model_path,
        rl_model_path=rl_model_path,
        svd_cache_path=svd_cache_path,
        summaries=sorted(summaries, key=lambda item: int(item["step"])),
        trajectory_summary=trajectory_summary,
        output_dir=output_dir,
    )
    report_path = output_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")

    combined = {
        "config_path": str(config_path),
        "base_model_path": str(base_model_path),
        "rl_model_path": str(rl_model_path),
        "svd_cache_path": str(svd_cache_path),
        "trajectory_summary": trajectory_summary,
        "summaries": sorted(summaries, key=lambda item: int(item["step"])),
    }
    write_json(output_dir / "combined_summary.json", combined)
    print(f"ANALYSIS_OUTPUT_DIR {output_dir}")
    print(f"REPORT_PATH {report_path}")
    for summary in summaries:
        checkpoint_name = Path(summary["checkpoint_path"]).name
        print(
            "SUMMARY",
            checkpoint_name,
            f"global_cosine={summary['global_cosine']:.6f}",
            f"layer_mean_cosine={summary['layer_mean_cosine']:.6f}",
            f"projection_energy_capture_ratio={summary['projection_energy_capture_ratio']:.6f}",
            f"positive_rate={summary['alignment_significance']['positive_rate']:.6f}",
        )


if __name__ == "__main__":
    main()
