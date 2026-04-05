#!/usr/bin/env python3
"""
pareto_plot.py — BitAdapt Pareto Front Generator (IEEE Format)
Separate figures for pretrained vs scratch, FP32 visible, clean lambda annotations.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
import argparse, os

# =============================================================================
# IEEE STYLE
# =============================================================================
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

CONFIGS = ['tiny', 'small', 'medium', 'base', 'large', 'xlarge']
CONFIG_LABELS = {
    'tiny': 'Tiny', 'small': 'Small', 'medium': 'Medium',
    'base': 'Base', 'large': 'Large', 'xlarge': 'XLarge'
}
CONFIG_COLORS = {
    'tiny':   '#1f77b4',
    'small':  '#ff7f0e',
    'medium': '#2ca02c',
    'base':   '#d62728',
    'large':  '#9467bd',
    'xlarge': '#8c564b',
}
CONFIG_MARKERS = {
    'tiny': 'o', 'small': 's', 'medium': '^',
    'base': 'd', 'large': 'p', 'xlarge': 'h'
}
SCENARIOS = ['ericsson', 'stecath']
SCENARIO_TITLES = {
    'ericsson': 'Ericsson Site',
    'stecath':  'Sainte-Catherine Site'
}

# Lambda values to annotate — only annotate first and last to keep it clean
LAMBDAS_TO_ANNOTATE = {0.1, 20.0}

# =============================================================================
# DATA LOADING
# =============================================================================
def load_best_from_json(json_path):
    try:
        with open(json_path) as f:
            data = json.load(f)
        if not data:
            return None
        best = min(
            data.values(),
            key=lambda e: e.get('val_loss', float('inf'))
            if e.get('val_loss') is not None else float('inf')
        )
        sr, en = best.get('sum_rate'), best.get('energy')
        if sr is None or en is None:
            return None
        return float(sr), float(en)
    except Exception:
        return None

def load_fp32_info(path):
    try:
        with open(path) as f:
            info = json.load(f)
        return float(info['sr']), float(info['en'])
    except Exception:
        return None

def collect_results(output_base):
    fp32  = {s: {} for s in SCENARIOS}
    quant = {s: {c: {'pretrained': [], 'scratch': []} for c in CONFIGS} for s in SCENARIOS}

    for scenario in SCENARIOS:
        for cfg in CONFIGS:
            # FP32 baseline
            fp32_info = Path(output_base) / f"fp32_{scenario}_{cfg}" / f"baseline_info_{scenario}_{cfg}.json"
            if fp32_info.exists():
                r = load_fp32_info(fp32_info)
                if r:
                    fp32[scenario][cfg] = r

            # Quantized results
            for init_mode in ['pretrained', 'scratch']:
                for run_dir in sorted(Path(output_base).glob(f"quant_{init_mode}_{scenario}_{cfg}_lam*")):
                    lam_str = run_dir.name.split('_lam')[-1]
                    try:
                        lam = float(lam_str)
                    except ValueError:
                        continue
                    json_files = list(run_dir.rglob("results_L_*.json"))
                    if not json_files:
                        continue
                    r = load_best_from_json(json_files[0])
                    if r:
                        quant[scenario][cfg][init_mode].append((r[0], r[1], lam))

                # Sort by lambda
                quant[scenario][cfg][init_mode].sort(key=lambda x: x[2])

    return fp32, quant

# =============================================================================
# PLOTTING
# =============================================================================
def get_xy(sr, en, metric, invert_axes):
    """Convert SR/Energy to plot coordinates based on metric and axis orientation."""
    ee = sr / (en + 1e-8)
    m_val = ee if metric == 'EE' else en
    if invert_axes:
        return m_val, sr   # x=metric, y=SR
    else:
        return sr, m_val   # x=SR, y=metric

def set_axis_labels(ax, metric, invert_axes, left_col):
    """Set axis labels cleanly."""
    sr_label    = r'Sum-Rate (bit/s/Hz)'
    ee_label    = r'Energy Efficiency (bit/s/Hz/$\mu$J)'
    en_label    = r'Energy Consumption ($\mu$J)'
    metric_label = ee_label if metric == 'EE' else en_label

    if invert_axes:
        ax.set_xlabel(metric_label)
        if left_col:
            ax.set_ylabel(sr_label)
    else:
        ax.set_xlabel(sr_label)
        if left_col:
            ax.set_ylabel(metric_label)

def annotate_lambda(ax, x, y, lam, color, invert_axes):
    """Annotate a single lambda value with a clean offset."""
    offset = (6, 4) if not invert_axes else (4, 6)
    ax.annotate(
        rf'$\lambda\!=\!{lam:.1f}$',
        (x, y),
        textcoords='offset points',
        xytext=offset,
        fontsize=8,
        color=color,
        alpha=0.85,
    )

def plot_single(fp32, quant, scenario, init_mode, metric, invert_axes, ax):
    """Plot one panel (one scenario, one init_mode)."""

    all_x, all_y = [], []

    for cfg in CONFIGS:
        color  = CONFIG_COLORS[cfg]
        marker = CONFIG_MARKERS[cfg]

        # ── FP32 Baseline ──
        if cfg in fp32[scenario]:
            fp32_sr, fp32_en = fp32[scenario][cfg]
            x_fp, y_fp = get_xy(fp32_sr, fp32_en, metric, invert_axes)
            ax.scatter(x_fp, y_fp, s=200, color=color, marker='*',
                       edgecolors='black', linewidths=0.8, zorder=8, alpha=0.9)
            all_x.append(x_fp)
            all_y.append(y_fp)

        # ── Quantized Points ──
        pts = quant[scenario][cfg][init_mode]
        if not pts:
            continue

        xs = [get_xy(p[0], p[1], metric, invert_axes)[0] for p in pts]
        ys = [get_xy(p[0], p[1], metric, invert_axes)[1] for p in pts]
        lams = [p[2] for p in pts]

        all_x.extend(xs)
        all_y.extend(ys)

        # Line connecting lambda sweep
        ax.plot(xs, ys, color=color, linestyle='-', linewidth=1.6,
                alpha=0.85, zorder=4)

        # Points
        ax.scatter(xs, ys, s=60, color=color, marker=marker,
                   edgecolors='black', linewidths=0.7, zorder=5)

        # Lambda annotations — only first and last
        for x, y, lam in zip(xs, ys, lams):
            if lam in LAMBDAS_TO_ANNOTATE:
                annotate_lambda(ax, x, y, lam, color, invert_axes)

    # ── Axis scale ──
    if invert_axes:
        ax.set_xscale('log')
    else:
        ax.set_yscale('log')

    # ── Auto zoom with padding ──
    if all_x and all_y:
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        if ax.get_xscale() == 'log':
            ax.set_xlim(min_x * 0.6, max_x * 2.0)
        else:
            mx = (max_x - min_x) * 0.08
            ax.set_xlim(min_x - mx, max_x + mx)

        if ax.get_yscale() == 'log':
            ax.set_ylim(min_y * 0.6, max_y * 2.0)
        else:
            my = (max_y - min_y) * 0.08
            ax.set_ylim(min_y - my, max_y + my)

    ax.grid(True, which='major', linestyle='--', alpha=0.5)
    ax.grid(True, which='minor', linestyle=':', alpha=0.25)


def build_legend_handles():
    """Build legend: one entry per config + FP32/Quantized markers."""
    handles = []
    for cfg in CONFIGS:
        handles.append(Line2D(
            [0], [0], color='none',
            marker=CONFIG_MARKERS[cfg],
            markerfacecolor=CONFIG_COLORS[cfg],
            markeredgecolor='black',
            markersize=8,
            label=CONFIG_LABELS[cfg]
        ))
    handles.append(Line2D([0], [0], color='none', label=' '))
    handles.append(Line2D(
        [0], [0], color='gray', linestyle='none',
        marker='*', markersize=11,
        markerfacecolor='gray', markeredgecolor='black',
        label=r'FP32 Baseline'
    ))
    handles.append(Line2D(
        [0], [0], color='black', linestyle='-',
        marker='o', markersize=7,
        markerfacecolor='black', markeredgecolor='black',
        label=r'Quantized ($\lambda$ sweep)'
    ))
    return handles


def make_figure(fp32, quant, init_mode, metric, invert_axes, save_path):
    """
    One figure = 1 init_mode × 2 scenarios side by side.
    """
    init_label = 'Pretrained Initialization' if init_mode == 'pretrained' else 'From-Scratch Initialization'
    metric_str = 'Energy Efficiency' if metric == 'EE' else 'Energy Consumption'
    axis_str   = 'Inverted' if invert_axes else 'Standard'

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        rf'Pareto Front — {init_label} — {metric_str} ({axis_str} axes)',
        fontsize=13, fontweight='bold', y=1.01
    )

    for i, (ax, scenario) in enumerate(zip(axes, SCENARIOS)):
        ax.set_title(SCENARIO_TITLES[scenario], fontweight='bold', pad=6)
        plot_single(fp32, quant, scenario, init_mode, metric, invert_axes, ax)
        set_axis_labels(ax, metric, invert_axes, left_col=(i == 0))

    # Single legend on right axes
    handles = build_legend_handles()
    axes[1].legend(
        handles=handles, loc='best',
        frameon=True, fancybox=False,
        edgecolor='black', framealpha=1.0
    )

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_base', default='/home/salaglo/links/scratch/BitAdapt_runs')
    parser.add_argument('--save_dir',    default='/home/salaglo/links/scratch/BitAdapt_runs/plots')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    fp32, quant = collect_results(args.output_base)

    # 2 init_modes × 2 metrics × 2 axis orientations = 8 figures
    for init_mode in ['pretrained', 'scratch']:
        for metric in ['EE', 'Energy']:
            for invert in [False, True]:
                axis_tag = 'inverted' if invert else 'standard'
                fname = f"{args.save_dir}/pareto_{init_mode}_{metric}_{axis_tag}.png"
                make_figure(fp32, quant, init_mode, metric, invert, fname)

    print("\nDone! 8 figures generated.")