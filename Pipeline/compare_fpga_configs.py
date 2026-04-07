"""
compare_fpga_configs.py — Comparaison des runs FPGA tiny / small / medium
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
    "tiny_fpga":   "Tiny  (emb=64,  4H, 4L, ~348K)",
    "small_fpga":  "Small (emb=128, 4H, 4L, ~825K)",
    "medium_fpga": "Medium (emb=192, 6H, 4L, ~1.83M)",
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
        if cfg in summaries:
            existing_best = max(summaries[cfg]["results"].values(),
                                key=lambda r: r["sr"])["sr"]
            new_best = max(data["results"].values(),
                           key=lambda r: r["sr"])["sr"]
            if new_best <= existing_best:
                print(f"  Skip {run_dir.name} — {cfg} déjà chargé avec meilleur SR")
                continue
        summaries[cfg] = data
        print(f"  Chargé : {cfg} ({run_dir.name})")
    return summaries


def plot_combined_pareto(summaries, save_dir):
    save_dir = Path(save_dir)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, xscale, title_suffix in zip(
        axes, ["log", "linear"],
        ["(log scale)", "(linear — BitAdapt region)"]
    ):
        for cfg, data in summaries.items():
            fp32_sr = data["fp32_stats"]["sr"]
            fp32_en = data["fp32_stats"]["en"]
            results = data["results"]
            color  = COLORS.get(cfg, "gray")
            marker = MARKERS.get(cfg, "o")
            label  = LABELS.get(cfg, cfg)

            ax.scatter(fp32_en, fp32_sr, marker="*", s=400,
                       color=color, zorder=5, label=f"{label} — 8-bit ref")

            lambdas = sorted(results.keys(), key=float)
            sr_vals = [results[l]["sr"] for l in lambdas]
            en_vals = [results[l]["en"] for l in lambdas]

            ax.plot(en_vals, sr_vals, color=color,
                    linewidth=1.5, linestyle="--", alpha=0.6)
            for i, lam in enumerate(lambdas):
                ax.scatter(en_vals[i], sr_vals[i], color=color,
                           marker=marker, s=180, zorder=4)
                ax.annotate(f"λ={lam}", (en_vals[i], sr_vals[i]),
                            xytext=(6, 4), textcoords="offset points",
                            fontsize=8, color=color)

        ax.set_xscale(xscale)
        if xscale == "linear":
            ax.set_xlim(0, 0.25)
        ax.set_xlabel("Energy (µJ)", fontsize=12)
        ax.set_ylabel("Sum Rate (bps/Hz)", fontsize=12)
        ax.set_title(f"Pareto Front {title_suffix}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("BitAdapt Mixed-Precision MIMO Precoder — FPGA Pareto Front",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / "pareto_combined.png", dpi=300)
    plt.close(fig)
    print("  Sauvegardé : pareto_combined.png")


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
            data     = summaries[cfg]
            fp32_ref = (data["fp32_stats"]["sr"] if metric == "sr"
                        else data["fp32_stats"]["en"])
            vals = [data["results"][lam][metric] / fp32_ref * 100
                    if lam in data["results"] else 0 for lam in lambdas]
            bars = ax.bar(x + offsets[j], vals, width,
                          label=LABELS.get(cfg, cfg),
                          color=COLORS.get(cfg, "gray"), alpha=0.85)
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
    print("  Sauvegardé : normalized_comparison.png")
    
def plot_bit_heatmap_continuous(summaries, results_dir, save_dir):
    save_dir    = Path(save_dir)
    results_dir = Path(results_dir)

    for cfg, data in summaries.items():
        lambdas = sorted(data["results"].keys(), key=float)

        # Trouve run_dir
        run_dir = None
        for d in results_dir.iterdir():
            sp = d / "summary.json"
            if not sp.exists():
                continue
            with open(sp) as f:
                s = json.load(f)
            if s.get("cfg") == cfg:
                if run_dir is None:
                    run_dir = d
                else:
                    old_best = max(json.load(open(run_dir/"summary.json"))
                                   ["results"].values(), key=lambda r: r["sr"])["sr"]
                    new_best = max(s["results"].values(), key=lambda r: r["sr"])["sr"]
                    if new_best > old_best:
                        run_dir = d
        if run_dir is None:
            continue

        # Charge bits float depuis results_L_{lam}.json — dernier epoch
        all_bw_w = {}
        all_bw_a = {}

        for lam in lambdas:
            results_file = (run_dir / "phase2_quant" / f"lambda_{lam}"
                            / f"results_L_{lam}.json")
            if not results_file.exists():
                continue
            with open(results_file) as f:
                res = json.load(f)

            # Prend le dernier epoch disponible
            epochs = sorted([k for k in res.keys() if k.isdigit()], key=int)
            if not epochs:
                continue
            last = res[epochs[-1]]

            all_bw_w[lam] = last.get("bit",     {})
            all_bw_a[lam] = last.get("bit_act", {})

        if not all_bw_w:
            print(f"  Aucun results_L trouvé pour {cfg}")
            continue

        first_lam     = list(all_bw_w.keys())[0]
        layers        = list(all_bw_w[first_lam].keys())
        lambdas_avail = [l for l in lambdas if l in all_bw_w]

        mat_w = np.zeros((len(lambdas_avail), len(layers)))
        mat_a = np.zeros((len(lambdas_avail), len(layers)))

        for i, lam in enumerate(lambdas_avail):
            for j, layer in enumerate(layers):
                mat_w[i, j] = all_bw_w[lam].get(layer, 0)
                mat_a[i, j] = all_bw_a[lam].get(layer, 0)

        # Labels lisibles
        layer_labels = []
        for l in layers:
            l2 = (l.replace("layers.", "L")
                   .replace(".attention",      ".attention")
                   .replace(".feed_forward.0", ".FF.0")
                   .replace(".feed_forward.3", ".FF.3")
                   .replace("embedding", "embedding")
                   .replace("output",    "output"))
            layer_labels.append(l2)

        fig, axes = plt.subplots(1, 2, figsize=(18, max(5, len(lambdas_avail)*0.8+2)))
        fig.suptitle(f"Mixed-Precision Bit Assignment — {LABELS.get(cfg, cfg)}",
                     fontsize=14, fontweight="bold")

        for ax, mat, title in zip(axes, [mat_w, mat_a],
                                   ["WEIGHTS", "ACTIVATIONS"]):
            vmin_local = max(1, mat.min() - 0.5)
            vmax_local = mat.max() + 0.5

            im = ax.imshow(mat, aspect="auto", cmap="YlOrRd",
                           vmin=vmin_local, vmax=vmax_local,
                           interpolation="nearest")

            ax.set_xticks(range(len(layers)))
            ax.set_xticklabels(layer_labels, rotation=45, ha="right", fontsize=9)
            ax.set_yticks(range(len(lambdas_avail)))
            ax.set_yticklabels([f"{l}" for l in lambdas_avail], fontsize=10)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Layer", fontsize=10)
            ax.set_ylabel("Penalty (λ)", fontsize=10)

            for i in range(len(lambdas_avail)):
                for j in range(len(layers)):
                    val = mat[i, j]
                    norm_val = (val - vmin_local) / (vmax_local - vmin_local + 1e-8)
                    txt_color = "white" if norm_val > 0.6 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, fontweight="bold", color=txt_color)

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Bit-width", fontsize=9)

        fig.tight_layout()
        fname = save_dir / f"bit_heatmap_{cfg}.png"
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Sauvegardé : bit_heatmap_{cfg}.png")     
def print_summary_table(summaries):
    print("\n" + "=" * 75)
    print(f"{'Config':<12} {'Lambda':<8} {'SR (bps/Hz)':<14} {'SR %':<10} "
          f"{'E (µJ)':<12} {'E %':<10}")
    print("=" * 75)
    for cfg, data in summaries.items():
        fp32_sr = data["fp32_stats"]["sr"]
        fp32_en = data["fp32_stats"]["en"]
        print(f"{cfg:<12} {'8-bit ref':<8} {fp32_sr:<14.2f} {'100%':<10} "
              f"{fp32_en:<12.4f} {'100%':<10}")
        for lam in sorted(data["results"].keys(), key=float):
            r    = data["results"][lam]
            sr_p = r["sr"] / fp32_sr * 100
            en_p = r["en"] / fp32_en * 100
            print(f"{'':12} {f'λ={lam}':<8} {r['sr']:<14.2f} "
                  f"{f'{sr_p:.1f}%':<10} {r['en']:<12.4f} {f'{en_p:.1f}%':<10}")
        print("-" * 75)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str,
                        default="/export/tmp/sala/results_fpga")
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    save_dir = Path(args.save_dir or args.results_dir)

    print("Chargement des summaries...")
    summaries = load_summaries(args.results_dir)

    if not summaries:
        print("Aucun summary.json trouvé.")
        return

    print(f"\n{len(summaries)} config(s) : {list(summaries.keys())}")
    print_summary_table(summaries)

    print("\nGénération des figures...")
    plot_combined_pareto(summaries, save_dir)
    plot_normalized_comparison(summaries, save_dir)
    plot_bit_heatmap_continuous(summaries, args.results_dir, save_dir)

    print("\nTerminé.")


if __name__ == "__main__":
    main()