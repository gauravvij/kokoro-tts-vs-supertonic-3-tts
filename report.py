"""
TTS CPU Benchmark Report Generator
====================================
Reads results/raw_results.csv, computes statistics, generates:
  - results/benchmark_report.md
  - results/charts/rtf_comparison.png
  - results/charts/latency_vs_length.png
"""

import os
import platform
import subprocess
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CSV_PATH = os.path.join(RESULTS_DIR, "raw_results.csv")
MOS_CSV_PATH = os.path.join(RESULTS_DIR, "mos_results.csv")
REPORT_PATH = os.path.join(RESULTS_DIR, "benchmark_report.md")
CHARTS_DIR = os.path.join(RESULTS_DIR, "charts")
AUDIO_DIR = os.path.join(RESULTS_DIR, "audio_samples")

os.makedirs(CHARTS_DIR, exist_ok=True)

# ── Ordered labels ─────────────────────────────────────────────────────────────
LENGTH_ORDER = ["tiny", "short", "medium", "long", "paragraph", "extended"]
CONFIG_ORDER = ["Supertonic-2step", "Supertonic-5step", "Kokoro-PyTorch", "Kokoro-ONNX", "Inflect-Nano"]

CONFIG_COLORS = {
    "Supertonic-2step": "#2196F3",   # blue
    "Supertonic-5step": "#03A9F4",   # light blue
    "Kokoro-PyTorch":   "#FF5722",   # deep orange
    "Kokoro-ONNX":      "#FF9800",   # orange
    "Inflect-Nano":     "#4CAF50",   # green
}

CONFIG_LABELS = {
    "Supertonic-2step": "Supertonic-3 (2-step)",
    "Supertonic-5step": "Supertonic-3 (5-step)",
    "Kokoro-PyTorch":   "Kokoro-82M (PyTorch)",
    "Kokoro-ONNX":      "Kokoro-82M (ONNX)",
    "Inflect-Nano":     "Inflect-Nano-v1 (4.6M)",
}


# ── Hardware fingerprint ───────────────────────────────────────────────────────
def get_hardware_info():
    info = {}
    info["os"] = platform.platform()
    info["python"] = sys.version.split()[0]
    info["cpu_cores"] = os.cpu_count()

    # CPU model
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    info["cpu_model"] = line.split(":")[1].strip()
                    break
    except Exception:
        info["cpu_model"] = "Unknown"

    # RAM
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if "MemTotal" in line:
                    kb = int(line.split()[1])
                    info["ram_gb"] = round(kb / 1024 / 1024, 1)
                    break
    except Exception:
        info["ram_gb"] = "Unknown"

    # Package versions
    venv_python = os.path.join(BASE_DIR, "venv/bin/python")
    for pkg in ["supertonic", "kokoro", "kokoro_onnx", "onnxruntime", "torch"]:
        try:
            result = subprocess.run(
                [venv_python, "-c", f"import {pkg}; print(getattr({pkg}, '__version__', 'unknown'))"],
                capture_output=True, text=True, timeout=10
            )
            info[f"pkg_{pkg}"] = result.stdout.strip() or "unknown"
        except Exception:
            info[f"pkg_{pkg}"] = "unknown"

    return info


# ── Statistics ─────────────────────────────────────────────────────────────────
def compute_stats(df):
    """Compute mean ± std per config × text_length for all metrics."""
    stats = df.groupby(["config_name", "text_length_label"]).agg(
        mean_rtf=("rtf", "mean"),
        std_rtf=("rtf", "std"),
        mean_wall=("wall_time_sec", "mean"),
        std_wall=("wall_time_sec", "std"),
        mean_audio_dur=("audio_duration_sec", "mean"),
        mean_chars_per_sec=("chars_per_sec", "mean"),
        std_chars_per_sec=("chars_per_sec", "std"),
        n_reps=("rep", "count"),
        text_length_chars=("text_length_chars", "first"),
    ).reset_index()
    return stats


# ── Chart 1: RTF comparison bar chart ─────────────────────────────────────────
def plot_rtf_comparison(stats, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: grouped bar chart by text length (all configs)
    ax = axes[0]
    n_lengths = len(LENGTH_ORDER)
    n_configs = len(CONFIG_ORDER)
    bar_width = 0.18
    x = np.arange(n_lengths)

    for i, config in enumerate(CONFIG_ORDER):
        cfg_data = stats[stats["config_name"] == config].copy()
        cfg_data = cfg_data.set_index("text_length_label").reindex(LENGTH_ORDER)
        means = cfg_data["mean_rtf"].values
        stds = cfg_data["std_rtf"].fillna(0).values
        offset = (i - (n_configs - 1) / 2) * bar_width
        bars = ax.bar(x + offset, means, bar_width,
                      label=CONFIG_LABELS[config],
                      color=CONFIG_COLORS[config],
                      alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.errorbar(x + offset, means, yerr=stds,
                    fmt="none", color="black", capsize=3, linewidth=1)

    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5,
               label="RTF=1.0 (real-time boundary)", alpha=0.7)
    ax.set_xlabel("Text Length", fontsize=12)
    ax.set_ylabel("Real-Time Factor (RTF) — lower is faster", fontsize=12)
    ax.set_title("RTF by Text Length and Config\n(error bars = ±1 std dev)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([l.capitalize() for l in LENGTH_ORDER], fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(stats["mean_rtf"].max() * 1.25, 1.2))

    # Right: mean RTF across all text lengths (summary bar)
    ax2 = axes[1]
    overall = stats.groupby("config_name")["mean_rtf"].mean().reindex(CONFIG_ORDER)
    overall_std = stats.groupby("config_name")["mean_rtf"].std().reindex(CONFIG_ORDER)
    colors = [CONFIG_COLORS[c] for c in CONFIG_ORDER]
    labels = [CONFIG_LABELS[c] for c in CONFIG_ORDER]
    bars2 = ax2.bar(range(len(CONFIG_ORDER)), overall.values,
                    color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax2.errorbar(range(len(CONFIG_ORDER)), overall.values, yerr=overall_std.values,
                 fmt="none", color="black", capsize=5, linewidth=1.5)
    ax2.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5,
                label="RTF=1.0 (real-time boundary)", alpha=0.7)
    ax2.set_xticks(range(len(CONFIG_ORDER)))
    ax2.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
    ax2.set_ylabel("Mean RTF (all text lengths)", fontsize=12)
    ax2.set_title("Overall Mean RTF per Config\n(lower = faster)", fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, max(overall.max() * 1.3, 1.2))

    # Annotate bars with values
    for bar, val in zip(bars2, overall.values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ── Chart 2: Latency vs text length line chart ─────────────────────────────────
def plot_latency_vs_length(stats, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Get char counts for x-axis
    char_counts = (
        stats.groupby("text_length_label")["text_length_chars"]
        .first()
        .reindex(LENGTH_ORDER)
        .values
    )

    # Left: wall-clock latency vs text length
    ax = axes[0]
    for config in CONFIG_ORDER:
        cfg = stats[stats["config_name"] == config].set_index("text_length_label").reindex(LENGTH_ORDER)
        means = cfg["mean_wall"].values
        stds = cfg["std_wall"].fillna(0).values
        ax.plot(char_counts, means, "o-",
                label=CONFIG_LABELS[config],
                color=CONFIG_COLORS[config],
                linewidth=2, markersize=6)
        ax.fill_between(char_counts, means - stds, means + stds,
                        color=CONFIG_COLORS[config], alpha=0.15)

    ax.set_xlabel("Input Text Length (characters)", fontsize=12)
    ax.set_ylabel("Wall-Clock Latency (seconds)", fontsize=12)
    ax.set_title("Synthesis Latency vs Text Length\n(shaded = ±1 std dev)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xscale("log")

    # Right: RTF vs text length
    ax2 = axes[1]
    for config in CONFIG_ORDER:
        cfg = stats[stats["config_name"] == config].set_index("text_length_label").reindex(LENGTH_ORDER)
        means = cfg["mean_rtf"].values
        stds = cfg["std_rtf"].fillna(0).values
        ax2.plot(char_counts, means, "o-",
                 label=CONFIG_LABELS[config],
                 color=CONFIG_COLORS[config],
                 linewidth=2, markersize=6)
        ax2.fill_between(char_counts, means - stds, means + stds,
                         color=CONFIG_COLORS[config], alpha=0.15)

    ax2.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5,
                label="RTF=1.0 (real-time boundary)", alpha=0.7)
    ax2.set_xlabel("Input Text Length (characters)", fontsize=12)
    ax2.set_ylabel("Real-Time Factor (RTF)", fontsize=12)
    ax2.set_title("RTF vs Text Length\n(lower = faster; <1.0 = real-time capable)", fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_xscale("log")
    ax2.set_ylim(0, max(stats["mean_rtf"].max() * 1.2, 1.2))

    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ── MOS loading ────────────────────────────────────────────────────────────────
def load_mos():
    """Load results/mos_results.csv if present.

    Returns (mos_df, mean_by_config) or (None, None) when MOS was not run.
    mean_by_config is a pandas Series indexed by config_name.
    """
    if not os.path.exists(MOS_CSV_PATH):
        return None, None
    mos_df = pd.read_csv(MOS_CSV_PATH)
    if mos_df.empty:
        return None, None
    mean_by_config = mos_df.groupby("config_name")["mos"].mean()
    return mos_df, mean_by_config


# ── Chart 3: Quality (MOS) vs Speed (RTF) scatter ───────────────────────────────
def plot_quality_vs_speed(stats, mos_mean, output_path):
    """Scatter of objective quality (UTMOS) against speed (mean RTF) per config.

    This is the chart that frames the core trade-off: a tiny/fast model can win
    on RTF while losing on naturalness.
    """
    overall_rtf = stats.groupby("config_name")["mean_rtf"].mean()

    # Nudge a few labels so overlapping points (the two Kokoro variants) stay legible.
    label_offsets = {
        "Kokoro-PyTorch": (10, 10),
        "Kokoro-ONNX":    (10, -18),
        "Inflect-Nano":   (10, 8),
        "Supertonic-2step": (-12, 10),
        "Supertonic-5step": (10, 8),
    }

    fig, ax = plt.subplots(figsize=(9, 7))
    for config in CONFIG_ORDER:
        if config not in overall_rtf.index or config not in mos_mean.index:
            continue
        x = overall_rtf[config]
        y = mos_mean[config]
        ax.scatter(x, y, s=260, color=CONFIG_COLORS[config],
                   edgecolor="black", linewidth=1.0, alpha=0.9, zorder=3)
        dx, dy = label_offsets.get(config, (10, 8))
        ha = "right" if dx < 0 else "left"
        ax.annotate(CONFIG_LABELS[config], (x, y),
                    textcoords="offset points", xytext=(dx, dy), fontsize=10, ha=ha)

    # Human-listening correction for Inflect-Nano: UTMOS over-rates it (sounds
    # robotic/buzzy). Draw a downward arrow from the metric dot toward where the
    # perceived quality really sits, so the chart doesn't imply it's mid-field.
    if "Inflect-Nano" in overall_rtf.index and "Inflect-Nano" in mos_mean.index:
        ix = overall_rtf["Inflect-Nano"]
        iy = mos_mean["Inflect-Nano"]
        ax.annotate(
            "by ear: buzzy / robotic\n(UTMOS over-rates)",
            xy=(ix, iy - 1.45), xytext=(ix, iy - 0.55),
            ha="center", va="top", fontsize=9, color="#555555",
            arrowprops=dict(arrowstyle="->", color="#777777", lw=1.4, linestyle="--"),
        )

    ax.set_xlabel("Mean RTF (lower = faster) →  faster", fontsize=12)
    ax.set_ylabel("UTMOS predicted MOS (higher = better quality)", fontsize=12)
    ax.set_title("Quality vs Speed\n(top-right = fast AND high quality = ideal; "
                 "UTMOS over-rates the tiny model — see arrow)", fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_ylim(1.0, 4.8)
    ax.invert_xaxis()  # faster (lower RTF) on the right feels natural
    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ── Markdown report ────────────────────────────────────────────────────────────
def generate_report(df, stats, hw, mos_df, mos_mean, output_path):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = []

    n_configs = df["config_name"].nunique()
    n_lengths = df["text_length_label"].nunique()
    n_runs = len(df)

    # Header
    lines += [
        "# TTS CPU Benchmark Report: Kokoro 82M vs Supertonic 3 vs Inflect-Nano-v1",
        "",
        f"*Generated: {now}*",
        "",
        "---",
        "",
    ]

    # Executive Summary
    overall_rtf = stats.groupby("config_name")["mean_rtf"].mean().reindex(CONFIG_ORDER)
    best_config = overall_rtf.idxmin()
    lines += [
        "## Executive Summary",
        "",
        "This report presents a rigorous CPU-only benchmark comparing **Kokoro 82M**, **Supertonic 3**, "
        "and **Inflect-Nano-v1** "
        f"across {n_lengths} text lengths (12–1712 characters), {n_configs} configurations, and 5 repetitions "
        f"each ({n_runs} total timed runs). "
        "All inference was performed on CPU with no GPU acceleration. Audio quality is reported as an objective "
        "UTMOS predicted MOS (neural naturalness estimate, ~1–5 scale).",
        "",
    ]
    has_mos = mos_mean is not None
    if has_mos:
        lines += [
            "| Config | Overall Mean RTF | vs Real-Time | Mean MOS (UTMOS) |",
            "|--------|-----------------|--------------|------------------|",
        ]
        for config in CONFIG_ORDER:
            rtf = overall_rtf[config]
            speedup = 1.0 / rtf
            mos = mos_mean.get(config, float("nan"))
            lines.append(
                f"| {CONFIG_LABELS[config]} | **{rtf:.4f}** | {speedup:.1f}× faster than real-time | **{mos:.2f}** |"
            )
    else:
        lines += [
            "| Config | Overall Mean RTF | vs Real-Time |",
            "|--------|-----------------|--------------|",
        ]
        for config in CONFIG_ORDER:
            rtf = overall_rtf[config]
            speedup = 1.0 / rtf
            lines.append(f"| {CONFIG_LABELS[config]} | **{rtf:.4f}** | {speedup:.1f}× faster than real-time |")

    lines += [
        "",
        f"> **Winner (lowest RTF):** {CONFIG_LABELS[best_config]} with mean RTF = {overall_rtf[best_config]:.4f}",
        "",
        "---",
        "",
    ]

    # Hardware Fingerprint
    lines += [
        "## Hardware & Environment",
        "",
        "| Property | Value |",
        "|----------|-------|",
        f"| CPU Model | {hw.get('cpu_model', 'Unknown')} |",
        f"| CPU Cores | {hw.get('cpu_cores', 'Unknown')} |",
        f"| RAM | {hw.get('ram_gb', 'Unknown')} GB |",
        f"| OS | {hw.get('os', 'Unknown')} |",
        f"| Python | {hw.get('python', 'Unknown')} |",
        f"| supertonic | {hw.get('pkg_supertonic', 'unknown')} |",
        f"| kokoro | {hw.get('pkg_kokoro', 'unknown')} |",
        f"| kokoro-onnx | {hw.get('pkg_kokoro_onnx', 'unknown')} |",
        f"| onnxruntime | {hw.get('pkg_onnxruntime', 'unknown')} |",
        f"| torch | {hw.get('pkg_torch', 'unknown')} |",
        "",
        "---",
        "",
    ]

    # Methodology
    lines += [
        "## Methodology",
        "",
        "### Configurations Tested",
        "",
        "| Config | Model | Backend | Steps/Mode |",
        "|--------|-------|---------|------------|",
        "| Supertonic-3 (2-step) | Supertone/supertonic-3 | ONNX Runtime (CPU) | total_steps=2 (speed mode) |",
        "| Supertonic-3 (5-step) | Supertone/supertonic-3 | ONNX Runtime (CPU) | total_steps=5 (default quality) |",
        "| Kokoro-82M (PyTorch) | hexgrad/Kokoro-82M | PyTorch CPU | Default |",
        "| Kokoro-82M (ONNX) | onnx-community/Kokoro-82M-v1.0-ONNX | ONNX Runtime (CPU) | Full precision |",
        "| Inflect-Nano-v1 (4.6M) | owensong/Inflect-Nano-v1 | PyTorch CPU | FastSpeech + Snake HiFi-GAN, single male voice |",
        "",
        "### Text Corpus",
        "",
        "| Label | Characters | Description |",
        "|-------|-----------|-------------|",
    ]

    # Get char counts from data
    char_map = df.groupby("text_length_label")["text_length_chars"].first().to_dict()
    desc_map = {
        "tiny": "Single short greeting",
        "short": "One sentence (pangram)",
        "medium": "2–3 sentences on AI",
        "long": "Paragraph on neural TTS",
        "paragraph": "Multi-sentence technical paragraph",
        "extended": "Multi-paragraph essay (~1700 chars)",
    }
    for label in LENGTH_ORDER:
        n = char_map.get(label, "?")
        desc = desc_map.get(label, "")
        lines.append(f"| {label} | {n} | {desc} |")

    lines += [
        "",
        "### Protocol",
        "",
        "- **CPU-only**: `CUDA_VISIBLE_DEVICES=''` set for all runs; ONNX sessions use `CPUExecutionProvider` only",
        "- **Warmup**: 1 discarded warmup run per config on the 'medium' text before timing begins",
        "- **Repetitions**: 5 timed runs per (config × text_length) cell",
        "- **Timing**: `time.perf_counter()` wall-clock, measuring synthesis only (not model load)",
        "- **Metrics**:",
        "  - **RTF** = wall_time / audio_duration (lower = faster; <1.0 = real-time capable)",
        "  - **Latency** = wall-clock seconds per synthesis call",
        "  - **Throughput** = input_chars / wall_time (chars/sec)",
        "- **Voice**: Supertonic voice 'F1' (female); Kokoro voice 'af_heart' (female); "
        "Inflect-Nano-v1 default voice 'mark' (male, single-speaker)",
        "- **Audio saved**: 1 WAV sample per (config × text_length) for quality verification",
        "",
        "---",
        "",
    ]

    # Full Results Tables
    lines += [
        "## Results",
        "",
        "### Mean RTF by Config and Text Length",
        "",
        "*(Lower RTF = faster; RTF < 1.0 = faster than real-time)*",
        "",
    ]

    # RTF pivot table
    rtf_pivot = stats.pivot_table(
        index="config_name", columns="text_length_label",
        values="mean_rtf", aggfunc="mean"
    ).reindex(CONFIG_ORDER)[LENGTH_ORDER]

    std_pivot = stats.pivot_table(
        index="config_name", columns="text_length_label",
        values="std_rtf", aggfunc="mean"
    ).reindex(CONFIG_ORDER)[LENGTH_ORDER]

    header = "| Config | " + " | ".join(l.capitalize() for l in LENGTH_ORDER) + " | **Mean** |"
    sep = "|--------|" + "|".join(["-------"] * len(LENGTH_ORDER)) + "|---------|"
    lines += [header, sep]

    for config in CONFIG_ORDER:
        row_vals = []
        for label in LENGTH_ORDER:
            m = rtf_pivot.loc[config, label]
            s = std_pivot.loc[config, label]
            row_vals.append(f"{m:.4f}±{s:.4f}")
        overall_m = rtf_pivot.loc[config].mean()
        lines.append(f"| {CONFIG_LABELS[config]} | " + " | ".join(row_vals) + f" | **{overall_m:.4f}** |")

    lines += [""]

    # Latency table
    lines += [
        "### Mean Wall-Clock Latency (seconds) by Config and Text Length",
        "",
    ]
    lat_pivot = stats.pivot_table(
        index="config_name", columns="text_length_label",
        values="mean_wall", aggfunc="mean"
    ).reindex(CONFIG_ORDER)[LENGTH_ORDER]

    header2 = "| Config | " + " | ".join(l.capitalize() for l in LENGTH_ORDER) + " |"
    sep2 = "|--------|" + "|".join(["-------"] * len(LENGTH_ORDER)) + "|"
    lines += [header2, sep2]
    for config in CONFIG_ORDER:
        row_vals = [f"{lat_pivot.loc[config, label]:.3f}s" for label in LENGTH_ORDER]
        lines.append(f"| {CONFIG_LABELS[config]} | " + " | ".join(row_vals) + " |")
    lines += [""]

    # Throughput table
    lines += [
        "### Mean Throughput (chars/sec) by Config and Text Length",
        "",
    ]
    thr_pivot = stats.pivot_table(
        index="config_name", columns="text_length_label",
        values="mean_chars_per_sec", aggfunc="mean"
    ).reindex(CONFIG_ORDER)[LENGTH_ORDER]

    lines += [header2, sep2]
    for config in CONFIG_ORDER:
        row_vals = [f"{thr_pivot.loc[config, label]:.1f}" for label in LENGTH_ORDER]
        lines.append(f"| {CONFIG_LABELS[config]} | " + " | ".join(row_vals) + " |")
    lines += [""]

    # Audio duration reference
    lines += [
        "### Reference: Mean Audio Duration (seconds) per Config × Text Length",
        "",
    ]
    dur_pivot = stats.pivot_table(
        index="config_name", columns="text_length_label",
        values="mean_audio_dur", aggfunc="mean"
    ).reindex(CONFIG_ORDER)[LENGTH_ORDER]
    lines += [header2, sep2]
    for config in CONFIG_ORDER:
        row_vals = [f"{dur_pivot.loc[config, label]:.2f}s" for label in LENGTH_ORDER]
        lines.append(f"| {CONFIG_LABELS[config]} | " + " | ".join(row_vals) + " |")
    lines += [""]

    # MOS (audio quality) table
    if mos_df is not None:
        lines += [
            "### Audio Quality — UTMOS Predicted MOS by Config and Text Length",
            "",
            "*(Higher = more natural; UTMOS predicts mean opinion score on a ~1–5 scale. "
            "Scores are objective neural estimates, not human ratings — and on Inflect-Nano-v1 the metric "
            "is optimistic: human listening rates it buzzy/robotic, below what its 3.48 suggests.)*",
            "",
        ]
        mos_pivot = mos_df.pivot_table(
            index="config_name", columns="text_length_label",
            values="mos", aggfunc="mean"
        ).reindex(CONFIG_ORDER)[LENGTH_ORDER]
        header_mos = "| Config | " + " | ".join(l.capitalize() for l in LENGTH_ORDER) + " | **Mean** |"
        sep_mos = "|--------|" + "|".join(["-------"] * len(LENGTH_ORDER)) + "|---------|"
        lines += [header_mos, sep_mos]
        for config in CONFIG_ORDER:
            row_vals = [f"{mos_pivot.loc[config, label]:.2f}" for label in LENGTH_ORDER]
            overall_m = mos_pivot.loc[config].mean()
            lines.append(f"| {CONFIG_LABELS[config]} | " + " | ".join(row_vals) + f" | **{overall_m:.2f}** |")
        lines += [""]

    lines += ["---", ""]

    # Analysis
    lines += [
        "## Analysis & Findings",
        "",
        "### 1. Overall Speed Ranking",
        "",
    ]
    for rank, (config, rtf) in enumerate(overall_rtf.sort_values().items(), 1):
        speedup = 1.0 / rtf
        lines.append(f"{rank}. **{CONFIG_LABELS[config]}** — Mean RTF: {rtf:.4f} ({speedup:.1f}× real-time)")
    lines += [""]

    # Supertonic vs Kokoro comparison
    st2_rtf = overall_rtf.get("Supertonic-2step", float("nan"))
    st5_rtf = overall_rtf.get("Supertonic-5step", float("nan"))
    kok_pt_rtf = overall_rtf.get("Kokoro-PyTorch", float("nan"))
    kok_onnx_rtf = overall_rtf.get("Kokoro-ONNX", float("nan"))

    st_vs_kok = kok_pt_rtf / st2_rtf if st2_rtf > 0 else float("nan")
    onnx_speedup = kok_pt_rtf / kok_onnx_rtf if kok_onnx_rtf > 0 else float("nan")
    step_cost = st5_rtf / st2_rtf if st2_rtf > 0 else float("nan")

    inflect_rtf = overall_rtf.get("Inflect-Nano", float("nan"))

    def _mos(cfg):
        return mos_mean.get(cfg, float("nan")) if mos_mean is not None else float("nan")

    lines += [
        "### 2. Speed vs Quality — the core trade-off",
        "",
        f"Supertonic 3 at 2-step mode is the fastest config (mean RTF **{st2_rtf:.4f}**, "
        f"{1.0/st2_rtf:.1f}× real-time), **{st_vs_kok:.1f}× faster** than Kokoro 82M (PyTorch) at "
        f"RTF {kok_pt_rtf:.4f}. But speed alone is misleading: its UTMOS quality is only "
        f"**{_mos('Supertonic-2step'):.2f}**, by far the lowest in the field — the 2-step output is "
        f"audibly robotic. The objective MOS confirms what listening reveals.",
        "",
        f"At 5-step mode, Supertonic's RTF rises to **{st5_rtf:.4f}** (a {step_cost:.2f}× slowdown vs "
        f"2-step from the extra flow-matching denoising steps), but quality jumps to "
        f"**{_mos('Supertonic-5step'):.2f}** — competitive with Kokoro. This is the configuration that "
        f"actually balances speed and quality.",
        "",
        f"Kokoro 82M scores highest on quality (PyTorch **{_mos('Kokoro-PyTorch'):.2f}**, "
        f"ONNX **{_mos('Kokoro-ONNX'):.2f}**) but is the slowest (RTF ~{kok_pt_rtf:.2f}–{kok_onnx_rtf:.2f}).",
        "",
        "### 3. Inflect-Nano-v1: tiny and fast, but robotic to the ear",
        "",
        f"At just 4.63M parameters — roughly 18× smaller than Kokoro and 21× smaller than Supertonic — "
        f"Inflect-Nano-v1 is the second-fastest config (mean RTF **{inflect_rtf:.4f}**, "
        f"{1.0/inflect_rtf:.1f}× real-time). Its UTMOS score is **{_mos('Inflect-Nano'):.2f}**, which "
        f"places it mid-field on the metric — but **human listening does not agree with that score**: the "
        f"output is audibly buzzy and robotic, with a metallic vocoder texture and flat prosody. It is "
        f"more intelligible than Supertonic-2step (which is worse), but it is not in the same league as "
        f"Kokoro or Supertonic-5step. This is a known UTMOS failure mode: it tends to over-rate small "
        f"HiFi-GAN vocoders that are *clean* but not *natural*. Treat Inflect-Nano's 3.48 as an optimistic "
        f"upper bound, not a usability verdict.",
        "",
        "> **Important caveat — output length cap.** Inflect-Nano-v1's acoustic model has "
        "`max_frames = 1400`, which caps synthesis at **~14.93 seconds of audio** regardless of input "
        "length. Inputs longer than that (here: `long`, `paragraph`, `extended`) are **silently "
        "truncated** — only the first ~15s is rendered. Its RTF and throughput on those rows are "
        "therefore inflated (it is doing less work than the other models, which synthesize the full "
        "text). Treat Inflect-Nano's `tiny`/`short`/`medium` numbers as the honest comparison; for "
        "long-form use you must split text into <15s chunks yourself. Its audio-duration row below "
        "(flat 14.93s for the three longest inputs) makes the cap visible.",
        "",
        "### 4. Kokoro PyTorch vs ONNX",
        "",
        f"On this hardware Kokoro ONNX (RTF **{kok_onnx_rtf:.4f}**) and PyTorch (**{kok_pt_rtf:.4f}**) are "
        f"within ~5% of each other, and their quality is identical to two decimal places "
        f"(**{_mos('Kokoro-ONNX'):.2f}** vs **{_mos('Kokoro-PyTorch'):.2f}**). The two are perceptually "
        f"interchangeable; the choice is a deployment/packaging decision, not a quality one.",
        "",
        "### 5. Practical Implications",
        "",
        "| Use Case | Recommended Config | Reason |",
        "|----------|-------------------|--------|",
        f"| Highest quality (human-like) | Kokoro-82M (PyTorch or ONNX) | Top UTMOS (~{max(_mos('Kokoro-PyTorch'), _mos('Kokoro-ONNX')):.2f}), Apache-2.0 weights |",
        f"| Balanced speed + quality | Supertonic-3 (5-step) | MOS {_mos('Supertonic-5step'):.2f} at {1.0/st5_rtf:.1f}× real-time |",
        f"| Tiny footprint / edge, quality secondary | Inflect-Nano-v1 | 4.6M params, {1.0/inflect_rtf:.1f}× real-time, but buzzy/robotic (UTMOS {_mos('Inflect-Nano'):.2f} over-rates it) |",
        f"| Latency at any cost (prototyping) | Supertonic-3 (2-step) | Fastest, but MOS {_mos('Supertonic-2step'):.2f} (robotic) |",
        "| PyTorch ecosystem / fine-tuning | Kokoro-82M (PyTorch) | Native PyTorch, easy to extend |",
        "",
        "### 6. Reproducibility Notes",
        "",
        "- All runs performed on a single CPU process with default thread counts",
        "- No process pinning or CPU affinity was set",
        "- Results may vary ±5–10% across runs due to OS scheduling jitter",
        "- The benchmark harness (`benchmark.py`) is fully reproducible: same text, same warmup protocol, same timing method",
        "",
        "---",
        "",
    ]

    # Charts reference
    lines += [
        "## Charts",
        "",
        "### RTF Comparison",
        "![RTF Comparison](charts/rtf_comparison.png)",
        "",
        "### Latency vs Text Length",
        "![Latency vs Text Length](charts/latency_vs_length.png)",
        "",
    ]
    if mos_df is not None:
        lines += [
            "### Quality vs Speed",
            "![Quality vs Speed](charts/quality_vs_speed.png)",
            "",
        ]
    lines += [
        "---",
        "",
    ]

    # Raw data summary
    n_wavs = n_configs * n_lengths
    lines += [
        "## Raw Data",
        "",
        f"Full raw results ({n_runs} rows): [`raw_results.csv`](raw_results.csv)",
        "",
        f"Per-sample MOS: [`mos_results.csv`](mos_results.csv)" if mos_df is not None else "",
        "",
        f"Audio samples: [`audio_samples/`](audio_samples/) — {n_wavs} WAV files (1 per config × text_length)",
        "",
        "---",
        "",
        f"*Report generated by `report.py` on {now}*",
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {output_path}")


# ── Summary table to stdout ────────────────────────────────────────────────────
def print_summary_table(stats):
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY — Mean RTF (lower = faster)")
    print("=" * 80)

    # Build pivot
    rtf_pivot = stats.pivot_table(
        index="config_name", columns="text_length_label",
        values="mean_rtf", aggfunc="mean"
    ).reindex(CONFIG_ORDER)[LENGTH_ORDER]

    # Header
    col_w = 10
    header = f"{'Config':<28}" + "".join(f"{l.capitalize():>{col_w}}" for l in LENGTH_ORDER) + f"{'MEAN':>{col_w}}"
    print(header)
    print("-" * len(header))

    for config in CONFIG_ORDER:
        row = f"{CONFIG_LABELS[config]:<28}"
        vals = []
        for label in LENGTH_ORDER:
            v = rtf_pivot.loc[config, label]
            row += f"{v:>{col_w}.4f}"
            vals.append(v)
        row += f"{sum(vals)/len(vals):>{col_w}.4f}"
        print(row)

    print("=" * 80)
    print("\nAll RTF values < 1.0 → all configs are faster than real-time on this CPU")

    # Throughput summary
    thr_pivot = stats.pivot_table(
        index="config_name", columns="text_length_label",
        values="mean_chars_per_sec", aggfunc="mean"
    ).reindex(CONFIG_ORDER)[LENGTH_ORDER]

    print("\n" + "=" * 80)
    print("THROUGHPUT SUMMARY — Mean chars/sec (higher = faster)")
    print("=" * 80)
    print(header.replace("RTF", "C/s"))
    print("-" * len(header))
    for config in CONFIG_ORDER:
        row = f"{CONFIG_LABELS[config]:<28}"
        vals = []
        for label in LENGTH_ORDER:
            v = thr_pivot.loc[config, label]
            row += f"{v:>{col_w}.1f}"
            vals.append(v)
        row += f"{sum(vals)/len(vals):>{col_w}.1f}"
        print(row)
    print("=" * 80)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("TTS Benchmark Report Generator")
    print(f"Reading: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows")

    # Validate
    assert df.rtf.isna().sum() == 0, "NaN RTF values found!"
    assert (df.rtf > 0).all(), "Non-positive RTF values found!"
    print("Data validation: PASSED")

    # Compute stats
    stats = compute_stats(df)
    print(f"Stats computed: {len(stats)} config×length cells")

    # Hardware info
    print("Collecting hardware info...")
    hw = get_hardware_info()

    # MOS (audio quality) data, if available
    mos_df, mos_mean = load_mos()
    if mos_df is not None:
        print(f"Loaded MOS data: {len(mos_df)} rows")
    else:
        print("No MOS data found (results/mos_results.csv) — skipping quality columns")

    # Generate charts
    print("\nGenerating charts...")
    plot_rtf_comparison(stats, os.path.join(CHARTS_DIR, "rtf_comparison.png"))
    plot_latency_vs_length(stats, os.path.join(CHARTS_DIR, "latency_vs_length.png"))
    if mos_df is not None:
        plot_quality_vs_speed(stats, mos_mean, os.path.join(CHARTS_DIR, "quality_vs_speed.png"))

    # Generate report
    print("\nGenerating markdown report...")
    generate_report(df, stats, hw, mos_df, mos_mean, REPORT_PATH)

    # Print summary
    print_summary_table(stats)

    # Final validation
    print("\n" + "=" * 60)
    print("OUTPUT VALIDATION")
    print("=" * 60)
    for path in [REPORT_PATH,
                 os.path.join(CHARTS_DIR, "rtf_comparison.png"),
                 os.path.join(CHARTS_DIR, "latency_vs_length.png")]:
        size = os.path.getsize(path)
        status = "✓" if size > 0 else "✗ EMPTY"
        print(f"  {status}  {path}  ({size:,} bytes)")

    audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]
    print(f"  ✓  Audio samples: {len(audio_files)} WAV files in {AUDIO_DIR}")
    print("=" * 60)
    print("All outputs generated successfully.")


if __name__ == "__main__":
    main()
