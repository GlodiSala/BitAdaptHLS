"""
compare_fpga_configs.py — Comparaison des runs FPGA micro / tiny_fpga / small_fpga
Lit les summary.json de chaque run et génère un Pareto front combiné

Usage:
  python compare_fpga_configs.py --results_dir /export/tmp/sala/results_fpga
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "tiny_fpga":   "steelblue",
    "small_fpga":  "darkorange",
    "medium_fpga": "green",
}
MARKERS = {
    "tiny_fpga":   "o",
    "small_fpga":  "^",
    "medium_fpga": "s",
}
LABELS = {
    "tiny_fpga":   "Tiny  (emb=64,  4L, ~200K)",
    "small_fpga":  "Small (emb=128, 4L, ~480K)",
    "medium_fpga": "Medium (emb=128, 6L, ~1.5M)",
}


def load_summaries(results_dir):
    results_dir = Path(results_dir)
    summaries = {}

    for run_dir in sorted(results_dir.iterdir()):
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            data = json.load(f)

        cfg = data.get("cfg")
        if not cfg:
            continue

        # Si plusieurs runs du meme cfg, garde le plus recent (dossier trié)
        # OU garde celui avec le meilleur sum rate
        if cfg in summaries:
            existing_best = max(
                summaries[cfg]["results"].values(),
                key=lambda r: r["sr"]
            )["sr"]
            new_best = max(
                data["results"].values(),
                key=lambda r: r["sr"]
            )["sr"]
            if new_best <= existing_best:
                print(f"  Skip {run_dir.name} — {cfg} deja charge avec meilleur SR")
                continue

        summaries[cfg] = data
        print(f"  Charge : {cfg} ({run_dir.name})")

    return summaries


def plot_combined_pareto(summaries, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 7))

    for cfg, data in summaries.items():
        fp32_sr = data["fp32_stats"]["sr"]
        fp32_en = data["fp32_stats"]["en"]
        results = data["results"]
        color   = COLORS.get(cfg, "gray")
        marker  = MARKERS.get(cfg, "o")
        label   = LABELS.get(cfg, cfg)

        ax.scatter(fp32_en, fp32_sr,
                   marker="*", s=400, color=color, zorder=5,
                   label=f"{label} — 8-bit baseline")

        lambdas = sorted(results.keys(), key=float)
        sr_vals = [results[l]["sr"] for l in lambdas]
        en_vals = [results[l]["en"] for l in lambdas]

        ax.plot(en_vals, sr_vals, color=color, linewidth=1.5,
                linestyle="--", alpha=0.5)

        for i, lam in enumerate(lambdas):
            ax.scatter(en_vals[i], sr_vals[i],
                       color=color, marker=marker, s=180, zorder=4)
            ax.annotate(f"λ={lam}",
                        (en_vals[i], sr_vals[i]),
                        xytext=(5, 4), textcoords="offset points",
                        fontsize=7, color=color)

    ax.set_xlabel("Energy (µJ)", fontsize=12)
    ax.set_ylabel("Sum Rate (bps/Hz)", fontsize=12)
    ax.set_title("Pareto Front — FPGA Transformer Configs (ReLU + RMSNorm)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=1)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_dir / "pareto_combined.png", dpi=300)
    plt.close(fig)
    print(f"  Sauvegardé : {save_dir / 'pareto_combined.png'}")


def plot_normalized_comparison(summaries, save_dir):
    save_dir = Path(save_dir)

    all_lambdas = set()
    for data in summaries.values():
        all_lambdas.update(data["results"].keys())
    lambdas = sorted(all_lambdas, key=float)

    cfgs    = list(summaries.keys())
    x       = np.arange(len(lambdas))
    width   = 0.25
    offsets = np.linspace(-width, width, len(cfgs))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Normalized Performance — FPGA Configs vs Lambda",
                 fontsize=13, fontweight="bold")

    for ax_idx, metric in enumerate(["sr", "en"]):
        ax = axes[ax_idx]
        for j, cfg in enumerate(cfgs):
            data    = summaries[cfg]
            fp32_ref = data["fp32_stats"]["sr"] if metric == "sr" else data["fp32_stats"]["en"]
            vals    = []
            for lam in lambdas:
                if lam in data["results"]:
                    vals.append(data["results"][lam][metric] / fp32_ref * 100)
                else:
                    vals.append(0)

            bars = ax.bar(x + offsets[j], vals, width,
                          label=LABELS.get(cfg, cfg),
                          color=COLORS.get(cfg, "gray"),
                          alpha=0.85)
            ax.bar_label(bars, fmt="%.0f%%", fontsize=7, padding=2)

        ax.axhline(100, color="black", linestyle="--", linewidth=1, label="8-bit ref")
        if metric == "sr":
            ax.axhline(90, color="red", linestyle=":", linewidth=1, label="−10% target")
            ax.set_ylabel("Sum Rate (% of 8-bit ref)")
            ax.set_title("Sum Rate Retention")
        else:
            ax.set_ylabel("Energy (% of 8-bit ref)")
            ax.set_title("Energy Reduction")

        ax.set_xticks(x)
        ax.set_xticklabels([f"λ={l}" for l in lambdas])
        ax.set_ylim(0, 120)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.4)

    fig.tight_layout()
    fig.savefig(save_dir / "normalized_comparison.png", dpi=300)
    plt.close(fig)
    print(f"  Sauvegardé : {save_dir / 'normalized_comparison.png'}")


def print_summary_table(summaries):
    print("\n" + "=" * 75)
    print(f"{'Config':<12} {'Lambda':<8} {'SR (bps/Hz)':<14} {'SR %':<10} {'E (µJ)':<12} {'E %':<10}")
    print("=" * 75)
    for cfg, data in summaries.items():
        fp32_sr = data["fp32_stats"]["sr"]
        fp32_en = data["fp32_stats"]["en"]
        print(f"{cfg:<12} {'8-bit ref':<8} {fp32_sr:<14.2f} {'100%':<10} {fp32_en:<12.4f} {'100%':<10}")
        for lam in sorted(data["results"].keys(), key=float):
            r    = data["results"][lam]
            sr_p = r["sr"] / fp32_sr * 100
            en_p = r["en"] / fp32_en * 100
            print(f"{'':12} {f'λ={lam}':<8} {r['sr']:<14.2f} {f'{sr_p:.1f}%':<10} {r['en']:<12.4f} {f'{en_p:.1f}%':<10}")
        print("-" * 75)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str,
                        default="/export/tmp/sala/results_fpga")
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    save_dir = args.save_dir or args.results_dir

    print("Chargement des summaries...")
    summaries = load_summaries(args.results_dir)

    if not summaries:
        print("Aucun summary.json trouvé.")
        return

    print(f"\n{len(summaries)} config(s) trouvée(s) : {list(summaries.keys())}")

    print_summary_table(summaries)

    print("\nGénération des figures...")
    plot_combined_pareto(summaries, save_dir)
    plot_normalized_comparison(summaries, save_dir)

    print("\nTerminé.")


if __name__ == "__main__":
    main()