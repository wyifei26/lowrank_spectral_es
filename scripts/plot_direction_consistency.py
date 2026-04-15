from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


COLORS = {
    "effective": (178, 74, 42),
    "m_state": (31, 106, 165),
    "grid": (230, 230, 230),
    "axis": (0, 0, 0),
    "text": (0, 0, 0),
    "mean": (255, 255, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot direction-consistency summaries from JSON outputs.")
    parser.add_argument("--analysis-dir", required=True, help="Directory containing direction_consistency*.json")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def project(value: float, y_min: float, y_max: float, plot_top: int, plot_bottom: int) -> int:
    if y_max <= y_min:
        return (plot_top + plot_bottom) // 2
    ratio = (value - y_min) / (y_max - y_min)
    return int(plot_bottom - ratio * (plot_bottom - plot_top))


def make_canvas(width: int = 1080, height: int = 680) -> tuple[Image.Image, ImageDraw.ImageDraw, tuple[int, int, int, int]]:
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    plot_left = 100
    plot_top = 60
    plot_right = width - 40
    plot_bottom = height - 90
    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=COLORS["axis"], width=2)
    return image, draw, (plot_left, plot_top, plot_right, plot_bottom)


def draw_grid(draw: ImageDraw.ImageDraw, *, y_min: float, y_max: float, plot_box: tuple[int, int, int, int]) -> None:
    plot_left, plot_top, plot_right, plot_bottom = plot_box
    for tick in np.linspace(0.0, 1.0, num=5):
        y_value = y_min + tick * (y_max - y_min)
        py = project(y_value, y_min, y_max, plot_top, plot_bottom)
        draw.line((plot_left, py, plot_right, py), fill=COLORS["grid"], width=1)
        draw.text((12, py - 8), f"{y_value:.4f}", fill=COLORS["text"])


def draw_summary_interval(
    draw: ImageDraw.ImageDraw,
    *,
    center_x: int,
    summary: dict[str, Any],
    color: tuple[int, int, int],
    y_min: float,
    y_max: float,
    plot_box: tuple[int, int, int, int],
) -> None:
    plot_left, plot_top, plot_right, plot_bottom = plot_box
    q10 = float(summary["q10"])
    q25 = float(summary["q25"])
    q75 = float(summary["q75"])
    q90 = float(summary["q90"])
    median = float(summary["median"])
    mean = float(summary["mean"])

    y_q10 = project(q10, y_min, y_max, plot_top, plot_bottom)
    y_q25 = project(q25, y_min, y_max, plot_top, plot_bottom)
    y_q75 = project(q75, y_min, y_max, plot_top, plot_bottom)
    y_q90 = project(q90, y_min, y_max, plot_top, plot_bottom)
    y_median = project(median, y_min, y_max, plot_top, plot_bottom)
    y_mean = project(mean, y_min, y_max, plot_top, plot_bottom)

    draw.line((center_x, y_q10, center_x, y_q90), fill=color, width=4)
    draw.rectangle((center_x - 18, y_q75, center_x + 18, y_q25), outline=color, fill=tuple(min(255, c + 40) for c in color))
    draw.line((center_x - 20, y_median, center_x + 20, y_median), fill=COLORS["axis"], width=2)
    draw.ellipse((center_x - 6, y_mean - 6, center_x + 6, y_mean + 6), outline=color, fill=COLORS["mean"], width=2)


def save_summary_chart(
    *,
    title: str,
    y_label: str,
    effective_summary: dict[str, Any],
    m_summary: dict[str, Any],
    output_path: Path,
) -> None:
    values = [
        float(effective_summary["q10"]),
        float(effective_summary["q90"]),
        float(m_summary["q10"]),
        float(m_summary["q90"]),
        float(effective_summary["mean"]),
        float(m_summary["mean"]),
    ]
    y_min = min(values)
    y_max = max(values)
    pad = max(1e-4, 0.08 * max(abs(y_min), abs(y_max), 1e-4))
    y_min -= pad
    y_max += pad

    image, draw, plot_box = make_canvas(width=920, height=620)
    plot_left, plot_top, plot_right, plot_bottom = plot_box
    draw.text((plot_left, 18), title, fill=COLORS["text"])
    draw.text((20, (plot_top + plot_bottom) // 2), y_label, fill=COLORS["text"])
    draw_grid(draw, y_min=y_min, y_max=y_max, plot_box=plot_box)

    centers = [plot_left + (plot_right - plot_left) // 3, plot_left + 2 * (plot_right - plot_left) // 3]
    draw_summary_interval(
        draw,
        center_x=centers[0],
        summary=effective_summary,
        color=COLORS["effective"],
        y_min=y_min,
        y_max=y_max,
        plot_box=plot_box,
    )
    draw_summary_interval(
        draw,
        center_x=centers[1],
        summary=m_summary,
        color=COLORS["m_state"],
        y_min=y_min,
        y_max=y_max,
        plot_box=plot_box,
    )
    draw.text((centers[0] - 40, plot_bottom + 20), "effective_M", fill=COLORS["text"])
    draw.text((centers[1] - 28, plot_bottom + 20), "m_state", fill=COLORS["text"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def save_series_plot(
    *,
    x_values: list[int],
    series: list[tuple[str, list[float], tuple[int, int, int]]],
    title: str,
    y_label: str,
    path: Path,
) -> None:
    width = 1180
    height = 720
    margin_left = 100
    margin_right = 40
    margin_top = 56
    margin_bottom = 96
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    all_points = [
        value
        for _, values, _ in series
        for value in values
        if value is not None and math.isfinite(value)
    ]
    if not x_values or not all_points:
        draw.text((20, 20), f"{title}: no finite data", fill=COLORS["text"])
        image.save(path)
        return

    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(all_points)
    max_y = max(all_points)
    if min_x == max_x:
        max_x += 1
    if min_y == max_y:
        delta = 1e-4 if min_y == 0.0 else abs(min_y) * 0.1
        min_y -= delta
        max_y += delta
    else:
        delta = 0.1 * max(abs(min_y), abs(max_y), 1e-4)
        min_y -= delta
        max_y += delta

    plot_left = margin_left
    plot_top = margin_top
    plot_right = width - margin_right
    plot_bottom = height - margin_bottom
    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=COLORS["axis"], width=2)
    draw.text((plot_left, 16), title, fill=COLORS["text"])
    draw.text((20, (plot_top + plot_bottom) // 2), y_label, fill=COLORS["text"])
    draw.text(((plot_left + plot_right) // 2, height - 28), "step", fill=COLORS["text"])

    def xy(step: int, value: float) -> tuple[int, int]:
        x_ratio = (step - min_x) / (max_x - min_x)
        y_ratio = (value - min_y) / (max_y - min_y)
        px = int(plot_left + x_ratio * (plot_right - plot_left))
        py = int(plot_bottom - y_ratio * (plot_bottom - plot_top))
        return px, py

    for tick in np.linspace(0.0, 1.0, num=5):
        y_value = min_y + tick * (max_y - min_y)
        _, py = xy(min_x, y_value)
        draw.line((plot_left, py, plot_right, py), fill=COLORS["grid"], width=1)
        draw.text((12, py - 8), f"{y_value:.4f}", fill=COLORS["text"])

    for step in x_values:
        px, _ = xy(step, min_y)
        draw.line((px, plot_top, px, plot_bottom), fill=(245, 245, 245), width=1)
        draw.text((px - 10, plot_bottom + 8), str(step), fill=COLORS["text"])

    legend_x = plot_left
    legend_y = plot_bottom + 34
    for label, values, color in series:
        points = [xy(step, value) for step, value in zip(x_values, values) if value is not None and math.isfinite(value)]
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        for px, py in points:
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color)
        draw.rectangle((legend_x, legend_y + 4, legend_x + 18, legend_y + 18), fill=color)
        draw.text((legend_x + 26, legend_y), label, fill=COLORS["text"])
        legend_x += max(180, 16 + 8 * len(label))

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def save_module_bar_chart(
    *,
    effective_means: dict[str, float],
    m_means: dict[str, float],
    output_path: Path,
) -> None:
    modules = sorted(m_means.keys())
    values = [float(effective_means[name]) for name in modules] + [float(m_means[name]) for name in modules] + [0.0]
    y_min = min(values)
    y_max = max(values)
    pad = max(1e-5, 0.2 * max(abs(y_min), abs(y_max), 1e-5))
    y_min -= pad
    y_max += pad

    image, draw, plot_box = make_canvas(width=1080, height=640)
    plot_left, plot_top, plot_right, plot_bottom = plot_box
    draw.text((plot_left, 18), "Mean Update-Direction Cosine by Module", fill=COLORS["text"])
    draw.text((20, (plot_top + plot_bottom) // 2), "mean cos(Δ_t, Δ_{t+1})", fill=COLORS["text"])
    draw_grid(draw, y_min=y_min, y_max=y_max, plot_box=plot_box)
    zero_y = project(0.0, y_min, y_max, plot_top, plot_bottom)
    draw.line((plot_left, zero_y, plot_right, zero_y), fill=COLORS["axis"], width=1)

    slot_width = (plot_right - plot_left) / max(1, len(modules))
    for idx, module in enumerate(modules):
        slot_left = plot_left + idx * slot_width
        center_x = int(slot_left + slot_width / 2)
        bar_half = max(8, int(slot_width * 0.16))
        eff_val = float(effective_means[module])
        m_val = float(m_means[module])
        eff_y = project(eff_val, y_min, y_max, plot_top, plot_bottom)
        m_y = project(m_val, y_min, y_max, plot_top, plot_bottom)
        draw.rectangle((center_x - 2 * bar_half, min(eff_y, zero_y), center_x - 4, max(eff_y, zero_y)), fill=COLORS["effective"])
        draw.rectangle((center_x + 4, min(m_y, zero_y), center_x + 2 * bar_half, max(m_y, zero_y)), fill=COLORS["m_state"])
        draw.text((center_x - 28, plot_bottom + 20), module, fill=COLORS["text"])

    draw.rectangle((plot_left, plot_bottom + 50, plot_left + 18, plot_bottom + 64), fill=COLORS["effective"])
    draw.text((plot_left + 26, plot_bottom + 46), "effective_M", fill=COLORS["text"])
    draw.rectangle((plot_left + 180, plot_bottom + 50, plot_left + 198, plot_bottom + 64), fill=COLORS["m_state"])
    draw.text((plot_left + 206, plot_bottom + 46), "m_state", fill=COLORS["text"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def save_extremes_chart(
    *,
    effective_payload: dict[str, Any],
    m_payload: dict[str, Any],
    output_path: Path,
) -> None:
    image, draw, plot_box = make_canvas(width=1180, height=680)
    plot_left, plot_top, plot_right, plot_bottom = plot_box
    draw.text((plot_left, 18), "Consecutive Update Cosine Extremes", fill=COLORS["text"])
    draw.text((20, (plot_top + plot_bottom) // 2), "cos(Δ_t, Δ_{t+1})", fill=COLORS["text"])

    eff_lows = [float(item["cosine"]) for item in effective_payload["lowest_delta_pairs"][:10]]
    eff_highs = [float(item["cosine"]) for item in effective_payload.get("highest_delta_pairs", [])[:10]]
    m_lows = [float(item["cosine"]) for item in m_payload["lowest_delta_pairs"][:10]]
    m_highs = [float(item["cosine"]) for item in m_payload.get("highest_delta_pairs", [])[:10]]
    values = eff_lows + eff_highs + m_lows + m_highs + [0.0]
    y_min = min(values)
    y_max = max(values)
    pad = max(1e-4, 0.2 * max(abs(y_min), abs(y_max), 1e-4))
    y_min -= pad
    y_max += pad
    draw_grid(draw, y_min=y_min, y_max=y_max, plot_box=plot_box)
    zero_y = project(0.0, y_min, y_max, plot_top, plot_bottom)
    draw.line((plot_left, zero_y, plot_right, zero_y), fill=COLORS["axis"], width=1)

    groups = [
        ("eff low", eff_lows, COLORS["effective"]),
        ("eff high", eff_highs, tuple(min(255, c + 35) for c in COLORS["effective"])),
        ("m low", m_lows, COLORS["m_state"]),
        ("m high", m_highs, tuple(min(255, c + 35) for c in COLORS["m_state"])),
    ]
    total_bars = sum(len(values_) for _, values_, _ in groups) + 6
    step = (plot_right - plot_left) / max(1, total_bars)
    cursor = plot_left + step
    for label, group_vals, color in groups:
        for value in group_vals:
            y = project(value, y_min, y_max, plot_top, plot_bottom)
            draw.rectangle((cursor - 8, min(y, zero_y), cursor + 8, max(y, zero_y)), fill=color)
            cursor += step
        draw.text((cursor - step * max(1, len(group_vals)) / 2 - 20, plot_bottom + 20), label, fill=COLORS["text"])
        cursor += 1.5 * step
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def format_stats(summary: dict[str, Any]) -> str:
    return (
        f"mean={float(summary['mean']):.6f}, "
        f"median={float(summary['median']):.6f}, "
        f"min={float(summary['min']):.6f}, "
        f"max={float(summary['max']):.6f}, "
        f"positive_rate={float(summary['positive_rate']):.3f}"
    )


def write_markdown_report(
    *,
    analysis_dir: Path,
    effective_payload: dict[str, Any],
    m_payload: dict[str, Any],
    sparse_payload: dict[str, Any],
) -> Path:
    report_path = analysis_dir / "direction_consistency_report.md"
    lines = [
        "# Direction Consistency Report",
        "",
        "## Interpretation",
        "",
        (
            "States are very smooth from one checkpoint to the next, but update directions are almost orthogonal on "
            "average. This means training follows a gradual trajectory through parameter/state space, while the actual "
            "step-to-step update direction is not strongly self-consistent."
        ),
        "",
        "## effective_M",
        "",
        f"- Consecutive state cosine: {format_stats(effective_payload['consecutive_state_cosine'])}",
        f"- Consecutive update delta cosine: {format_stats(effective_payload['consecutive_update_delta_cosine'])}",
        "",
        "## m_state",
        "",
        f"- Consecutive state cosine: {format_stats(m_payload['consecutive_state_cosine'])}",
        f"- Consecutive update delta cosine: {format_stats(m_payload['consecutive_update_delta_cosine'])}",
        "",
        "## Sparse 10-step View",
        "",
        f"- Sparse state cosine mean={float(sparse_payload['state_mean']):.6f}, min={float(sparse_payload['state_min']):.6f}",
        (
            f"- Sparse delta cosine mean={float(sparse_payload['delta_mean']):.6f}, "
            f"median={float(sparse_payload['delta_median']):.6f}, "
            f"positive_rate={float(sparse_payload['delta_positive_rate']):.3f}"
        ),
        "",
        "## Figures",
        "",
        "- `direction_consistency_state_cosine.png`",
        "- `direction_consistency_delta_cosine.png`",
        "- `direction_consistency_module_delta_bar.png`",
        "- `direction_consistency_sparse10_state.png`",
        "- `direction_consistency_sparse10_delta.png`",
        "- `direction_consistency_delta_extremes.png`",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir).resolve()
    effective_payload = load_json(analysis_dir / "direction_consistency.json")
    m_payload = load_json(analysis_dir / "direction_consistency_m_state.json")
    sparse_payload = load_json(analysis_dir / "direction_consistency_m_state_sparse10.json")

    save_summary_chart(
        title="Consecutive State Cosine Summary",
        y_label="cos(state_t, state_{t+1})",
        effective_summary=effective_payload["consecutive_state_cosine"],
        m_summary=m_payload["consecutive_state_cosine"],
        output_path=analysis_dir / "direction_consistency_state_cosine.png",
    )
    save_summary_chart(
        title="Consecutive Update Direction Cosine Summary",
        y_label="cos(Δ_t, Δ_{t+1})",
        effective_summary=effective_payload["consecutive_update_delta_cosine"],
        m_summary=m_payload["consecutive_update_delta_cosine"],
        output_path=analysis_dir / "direction_consistency_delta_cosine.png",
    )
    save_module_bar_chart(
        effective_means={key: float(value) for key, value in effective_payload["module_delta_cosine_mean"].items()},
        m_means={key: float(value) for key, value in m_payload["module_delta_cosine_mean"].items()},
        output_path=analysis_dir / "direction_consistency_module_delta_bar.png",
    )
    save_series_plot(
        x_values=[int(item["to_step"]) for item in sparse_payload["state_pairs"]],
        series=[("sparse10 state cosine", [float(item["cosine"]) for item in sparse_payload["state_pairs"]], COLORS["m_state"])],
        title="Sparse-10 State Cosine in m_state Space",
        y_label="cos(state_a, state_b)",
        path=analysis_dir / "direction_consistency_sparse10_state.png",
    )
    save_series_plot(
        x_values=[int(item["via_step"]) for item in sparse_payload["delta_pairs"]],
        series=[("sparse10 delta cosine", [float(item["cosine"]) for item in sparse_payload["delta_pairs"]], COLORS["effective"])],
        title="Sparse-10 Update Direction Cosine in m_state Space",
        y_label="cos(Δ_window_a, Δ_window_b)",
        path=analysis_dir / "direction_consistency_sparse10_delta.png",
    )
    save_extremes_chart(
        effective_payload=effective_payload,
        m_payload=m_payload,
        output_path=analysis_dir / "direction_consistency_delta_extremes.png",
    )
    report_path = write_markdown_report(
        analysis_dir=analysis_dir,
        effective_payload=effective_payload,
        m_payload=m_payload,
        sparse_payload=sparse_payload,
    )
    print(json.dumps({"analysis_dir": str(analysis_dir), "report_path": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
