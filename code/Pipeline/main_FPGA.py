import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

import Pipeline.process as process
import quan
import util
from Pipeline.Energy import analyze_and_compute_energy
from Pipeline.Learner import Learner
from Pipeline.Transformer_FPGA import StackedTransformer
from Pipeline.input_args import input_args
from Pipeline.utils import utils_
from main_v3 import (
    log_bitlengths, save_result_data, save_quantized_model, resolve_path
)

TRANSFORMER_CONFIGS = {
    "tiny_fpga":  dict(num_layers=4, embedding_dim=64,  num_heads=4, hidden_dim=512,  dropout=0.0),
    "small_fpga": dict(num_layers=4, embedding_dim=128, num_heads=4, hidden_dim=512,  dropout=0.0),
   "medium_fpga": dict(num_layers=4, embedding_dim=192, num_heads=6, hidden_dim=768, dropout=0.0)
}

DEFAULT_LAMBDAS = [1.0, 5.0, 20.0]

def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--transformer_cfg", type=str, default="tiny_fpga",
                        choices=list(TRANSFORMER_CONFIGS.keys()))
    parser.add_argument("--scenario",        type=str, default="stecath")
    parser.add_argument("--lambda_reg",      type=float, nargs="+", default=DEFAULT_LAMBDAS)
    parser.add_argument("--pretrain_epochs", type=int, default=100)
    parser.add_argument("--quant_epochs",    type=int, default=80)
    parser.add_argument("--output_dir",      type=str, default=None)
    parser.add_argument("--run_name",        type=str, default=None)

    overrides, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining  # ← GARDER CETTE LIGNE
    return overrides

def plot_pretrain_curve(sr_history, save_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(sr_history) + 1), sr_history, linewidth=2, color="steelblue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Sum Rate (bps/Hz)")
    ax.set_title("FP32 Pretraining — Sum Rate", fontweight="bold")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_quant_training(json_path, save_dir, lam, fp32_sr, fp32_en):
    with open(json_path) as f:
        data = json.load(f)

    epochs   = sorted([int(e) for e in data.keys()])
    sr       = [data[str(e)]["sum_rate"] for e in epochs]
    energy   = [data[str(e)]["energy"]   for e in epochs]
    val_loss = [data[str(e)].get("val_loss", None) for e in epochs]

    bit_layers = list(data[str(epochs[0])]["bit"].keys()) if data[str(epochs[0])]["bit"] else []
    bits_over_time = {
        k: [data[str(e)]["bit"].get(k, None) for e in epochs]
        for k in bit_layers
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Quantization Training — λ={lam}", fontsize=13, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(epochs, [s / fp32_sr * 100 for s in sr], color="steelblue", linewidth=2)
    ax.axhline(100, color="black", linestyle="--", linewidth=1, label="FP32 ref")
    ax.axhline(90,  color="red",   linestyle=":",  linewidth=1, label="10% target")
    ax.set_ylabel("Sum Rate (% of FP32)")
    ax.set_title("Sum Rate Retention")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, [e / fp32_en * 100 for e in energy], color="darkorange", linewidth=2)
    ax.axhline(100, color="black", linestyle="--", linewidth=1, label="8-bit ref")
    ax.set_ylabel("Energy (% of 8-bit ref)")
    ax.set_title("Energy Reduction")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    vl_clean = [v for v in val_loss if v is not None]
    ep_clean = [e for e, v in zip(epochs, val_loss) if v is not None]
    ax.plot(ep_clean, vl_clean, color="purple", linewidth=2)
    ax.set_ylabel("val_loss")
    ax.set_title("Normalized Loss")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(bit_layers), 1)))
    for (layer, vals), color in zip(bits_over_time.items(), cmap):
        clean_vals = [v for v in vals if v is not None]
        clean_eps  = [e for e, v in zip(epochs, vals) if v is not None]
        ax.plot(clean_eps, clean_vals, linewidth=1.5,
                label=layer.split(".")[-1], color=color)
    ax.set_ylabel("Bit-width")
    ax.set_title("Bit-width Evolution per Layer")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(Path(save_dir) / f"training_lambda_{lam}.png", dpi=300)
    plt.close(fig)

def plot_final_bits(quantizers, save_path):
    names  = list(quantizers.keys())
    w_bits = [math.ceil(q["weight"].bit.item()) if hasattr(q["weight"], "bit") else 16
              for q in quantizers.values()]
    a_bits = [math.ceil(q["activation"].bit.item()) if hasattr(q["activation"], "bit") else 16
              for q in quantizers.values()]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.8), 5))
    ax.bar(x - 0.2, w_bits, 0.4, label="Weights", color="steelblue")
    ax.bar(x + 0.2, a_bits, 0.4, label="Activations", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels([n.split(".")[-1] for n in names], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Bit-width")
    ax.set_title("Final Mixed-Precision Bit Assignment", fontweight="bold")
    ax.axhline(8, color="gray", linestyle="--", linewidth=1, label="8-bit uniform")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_pareto(fp32_stats, global_stats, save_dir):
    lambdas = sorted(global_stats.keys())
    sr_vals = [global_stats[l]["sr"] for l in lambdas]
    en_vals = [global_stats[l]["en"] for l in lambdas]
    colors  = plt.cm.viridis(np.linspace(0.2, 0.85, len(lambdas)))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(fp32_stats["en"], fp32_stats["sr"],
               marker="*", s=500, color="black", zorder=5, label="8-bit baseline")
    for i, lam in enumerate(lambdas):
        ax.scatter(en_vals[i], sr_vals[i], color=colors[i], s=220,
                   marker="o", label=f"BitAdapt λ={lam}", zorder=4)
        ax.annotate(f"λ={lam}", (en_vals[i], sr_vals[i]),
                    xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Energy (µJ)", fontsize=12)
    ax.set_ylabel("Sum Rate (bps/Hz)", fontsize=12)
    ax.set_title("Hardware Pareto Front — FPGA Transformer", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(Path(save_dir) / "pareto_front.png", dpi=300)
    plt.close(fig)

    x     = np.arange(len(lambdas))
    width = 0.35
    sr_n  = [s / fp32_stats["sr"] * 100 for s in sr_vals]
    en_n  = [e / fp32_stats["en"] * 100 for e in en_vals]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Normalized Performance vs Lambda", fontsize=13, fontweight="bold")

    ax = axes[0]
    bars = ax.bar(x, sr_n, width * 2, color="steelblue")
    ax.axhline(100, color="black", linestyle="--", linewidth=1, label="8-bit ref")
    ax.axhline(90,  color="red",   linestyle=":",  linewidth=1, label="10% target")
    ax.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"λ={l}" for l in lambdas])
    ax.set_ylabel("Sum Rate (% of 8-bit ref)")
    ax.set_title("Sum Rate Retention")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.4)
    ax.set_ylim(0, 115)

    ax = axes[1]
    bars = ax.bar(x, en_n, width * 2, color="darkorange")
    ax.axhline(100, color="black", linestyle="--", linewidth=1, label="8-bit ref")
    ax.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"λ={l}" for l in lambdas])
    ax.set_ylabel("Energy (% of 8-bit ref)")
    ax.set_title("Energy Reduction")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.4)
    ax.set_ylim(0, 115)

    fig.tight_layout()
    fig.savefig(Path(save_dir) / "normalized_perf.png", dpi=300)
    plt.close(fig)

def dump_hw_analysis(model, quantizers, fp32_sr, fp32_en, best_sr, best_en, save_path):
    hw = {
        "sr_fp32_ref":        fp32_sr,
        "en_8bit_ref_uJ":     fp32_en,
        "sr_best":            best_sr,
        "en_best_uJ":         best_en,
        "sr_degradation_pct": round((1 - best_sr / fp32_sr) * 100, 2),
        "en_reduction_pct":   round((1 - best_en / fp32_en) * 100, 2),
        "bit_widths":         {},
        "param_counts":       {},
        "total_params":       0,
    }

    total = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            n = module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
            hw["param_counts"][name] = n
            total += n
    hw["total_params"] = total

    for name, q in quantizers.items():
        w_bit = math.ceil(q["weight"].bit.item())     if hasattr(q["weight"],     "bit") else 16
        a_bit = math.ceil(q["activation"].bit.item()) if hasattr(q["activation"], "bit") else 16
        hw["bit_widths"][name] = {"weight": w_bit, "activation": a_bit}

    hw["hls_summary"] = {
    name: {
        "weight_format": f"ap_int<{v['weight']}>",
        "act_format":    f"ap_int<{v['activation']}>",
        "note":          "Delayed Scaling: acc_int * s_w * s_act + bias"
    }
    for name, v in hw["bit_widths"].items()
}

    with open(save_path, "w") as f:
        json.dump(hw, f, indent=2)
    return hw

def main():
    # ← APPEL UNIQUE ici, AVANT tout
    overrides = parse_args()
    # parse_args() fait sys.argv = [argv[0]] + remaining
    # remaining contient --scenario stecath pour input_args()

    cfg_tag      = overrides.transformer_cfg
    scenario_tag = overrides.scenario
    tcfg         = TRANSFORMER_CONFIGS[cfg_tag]
    lambdas      = overrides.lambda_reg

    # run_name et run_dir
    if overrides.run_name:
        run_name = overrides.run_name
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_name  = f"FPGA_{cfg_tag}_{scenario_tag}_{timestamp}"

    if overrides.output_dir:
        run_dir = Path(overrides.output_dir) / run_name
    else:
        run_dir = Path(__file__).parent / "results_fpga" / run_name

    # ← SUPPRIME tous les blocs DEBUG et le deuxième parse_args()
    # ← SUPPRIME script_dir — remplace par Path(__file__).parent

    run_dir.mkdir(parents=True, exist_ok=True)
    phase1_dir = run_dir / "phase1_pretrain"
    phase2_dir = run_dir / "phase2_quant"
    plots_dir  = run_dir / "plots"
    hls_dir    = run_dir / "hls_analysis"
    for d in [phase1_dir, phase2_dir, plots_dir, hls_dir]:
        d.mkdir(exist_ok=True)

    fp32_weights_path = phase1_dir / "fp32_best.pth"
    fp32_info_path    = phase1_dir / "baseline_info.json"

    # Config depuis le bon chemin
    CONFIG_PATH = Path(__file__).parent / "configs" / "base_config.yaml"
    args        = util.get_config(str(CONFIG_PATH))
    args.name   = run_name

    # Logger sans timestamp
    log_file = run_dir / f"{run_name}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ]
    )
    logger = logging.getLogger()

    logger.info(f"run_name   = {run_name}")
    logger.info(f"run_dir    = {run_dir}")
    logger.info(f"lambdas    = {lambdas}")
    logger.info(f"output_dir = {overrides.output_dir}")

    # input_args() — sys.argv contient encore --scenario stecath
    # grace au parse_known_args() + sys.argv = remaining dans parse_args()
    input_args_        = input_args()
    arguments          = input_args_.args
    arguments.scenario = [scenario_tag]


    if not torch.cuda.is_available():
        args.device.gpu  = []
        args.device.type = torch.device("cpu")
    else:
        args.device.gpu  = [0]
        args.device.type = torch.device("cuda:0")
        torch.cuda.set_device(0)

    Func = utils_(arguments)
    train_loader, test_loader = Func.Data_Load()
    val_loader = test_loader

    base_batch = 1000
    lr_scale   = min(arguments.batch_size / base_batch, 2.0)
    lr_model   = 1e-3

    # =========================================================================
    # PHASE 1 — FP32 Pretraining
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info(" PHASE 1 — FP32 Pretraining")
    logger.info("=" * 70)

    if fp32_weights_path.exists() and fp32_info_path.exists():
        logger.info("Phase 1 deja completee, chargement des resultats existants.")
    else:
        model    = StackedTransformer(**tcfg).to(args.device.type)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Modele : {cfg_tag} — {n_params:,} parametres")

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr_model * lr_scale,
            weight_decay=1e-5, amsgrad=True
        )
        n_epochs  = overrides.pretrain_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=1e-6
        )
        Learner_  = Learner(arguments)

        def criterion_fp32(FDP, channel, *args, **kwargs):
            return Learner_.criterium_FDP(FDP, channel)

        best_sr    = 0.0
        sr_history = []

        for epoch in range(1, n_epochs + 1):
            process.train(
                train_loader, model, criterion_fp32,
                optimizer, None, epoch, [], args, arguments, {}
            )
            sr, _, _ = process.validate(
                val_loader, model, criterion_fp32,
                epoch, [], args, arguments
            )
            scheduler.step()
            sr_history.append(float(sr))

            if sr > best_sr:
                best_sr = sr
                torch.save(model.state_dict(), fp32_weights_path)

            if epoch % 10 == 0:
                logger.info(
                    f"  Ep {epoch:3d}/{n_epochs} | SR={sr:.4f} "
                    f"(best={best_sr:.4f}) | LR={scheduler.get_last_lr()[0]:.2e}"
                )
        n_quant = sum(
            1 for m in model.modules()
            if isinstance(m, (torch.nn.Linear, torch.nn.MultiheadAttention))
            and not isinstance(m, torch.nn.modules.linear.NonDynamicallyQuantizableLinear)
        )
        Q8      = [8] * n_quant
        en_8bit = analyze_and_compute_energy(
            model, Q8, Q8, input_size=(1, 64, 128)
        ) * 1e-6

        baseline = {
            "sr":              best_sr,
            "en":              en_8bit,
            "n_params":        n_params,
            "n_quant_layers":  n_quant,
        }
        with open(fp32_info_path, "w") as f:
            json.dump(baseline, f, indent=2)

        plot_pretrain_curve(sr_history, plots_dir / "pretrain_curve.png")
        logger.info(
            f"Phase 1 terminee — SR={best_sr:.4f}, "
            f"E_8bit={en_8bit:.4f}uJ ({n_quant} couches)"
        )


    with open(fp32_info_path) as f:
        info = json.load(f)
    fp32_stats = {"sr": info["sr"], "en": info["en"],
                  "ee": info["sr"] / (info["en"] + 1e-8)}
    logger.info(f"Baseline — SR={fp32_stats['sr']:.4f}, E_8bit={fp32_stats['en']:.4f}uJ")

    # =========================================================================
    # PHASE 2 — Quantization sweep
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info(f" PHASE 2 — Quantization sweep  λ={lambdas}")
    logger.info("=" * 70)

    global_stats = {}

    for current_lambda in lambdas:
        logger.info(f" λ = {current_lambda}")

        lambda_dir = phase2_dir / f"lambda_{current_lambda}"
        file_path  = str(lambda_dir / f"results_L_{current_lambda}.json")

        # ── SKIP si déjà calculé ──────────────────────────────────────────
        if lambda_dir.exists() and Path(file_path).exists():
            logger.info(f"  λ={current_lambda} déjà calculé — skip.")
            with open(file_path) as f:
                data = json.load(f)
            best_epoch = min(data.keys(), key=lambda k: data[k]["val_loss"])
            best = data[best_epoch]
            global_stats[current_lambda] = {
                "sr": best["sum_rate"],
                "en": best["energy"],
                "ee": best["EE"],
            }
            continue
        # ─────────────────────────────────────────────────────────────────

        lambda_dir.mkdir(exist_ok=True)
        checkpoint_dir = lambda_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        model = StackedTransformer(**tcfg).to(args.device.type)
        model.load_state_dict(torch.load(fp32_weights_path, map_location=args.device.type))
        logger.info("  Poids FP32 charges")

        Learner_ = Learner(arguments,
                           fp32_energy_ref=fp32_stats["en"],
                           fp32_sr_ref=fp32_stats["sr"])

        replaced_modules, quantizers = quan.find_modules_to_quantize(model, args.quan)
        model      = quan.replace_module_by_names(model, replaced_modules)
        quantizers = quan.fix_quantizer_references(model, quantizers)
        quantizers = quan.calibrate_activation_quantizers(
            model, quantizers, train_loader, args.device.type, num_batches=10
        )

        bit_params   = [p for n, p in model.named_parameters() if "bit" in n.split(".")[-1]]
        other_params = [p for n, p in model.named_parameters() if "bit" not in n.split(".")[-1]]
        optimizer = torch.optim.AdamW([
            {"params": bit_params,   "lr": args.lr_bit * lr_scale,
             "weight_decay": arguments.wd, "amsgrad": True},
            {"params": other_params, "lr": lr_model * lr_scale,
             "weight_decay": 0.001,        "amsgrad": True},
        ])
        quant_epochs = overrides.quant_epochs
        scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=quant_epochs, eta_min=1e-6
        )

        def make_criterion(lam, learner):
            def criterion_wrapper(FDP, channel, *args, **kwargs):
                return learner.criterium_with_bitpruning(FDP, channel, *args, lambda_reg=lam)
            return criterion_wrapper

        criterion_quant = make_criterion(current_lambda, Learner_)

        def criterion_val(FDP, channel, *args, **kwargs):
            return Learner_.criterium_FDP(FDP, channel)

        result_log = {"bit": {}, "bit_act": {}, "sum_rate": None,
                      "EE": None, "energy": None, "val_loss": None}
        best_loss = float("inf")
        best_sr = best_en = best_ee = 0.0

        for epoch in range(1, quant_epochs + 1):
            process.train(
                train_loader, model, criterion_quant,
                optimizer, None, epoch, [], args, arguments, quantizers
            )
            sum_rate, _, _ = process.validate(
                val_loader, model, criterion_val,
                epoch, [], args, arguments
            )
            scheduler.step()

            result_log = log_bitlengths(quantizers, result_log)
            result_log["sum_rate"] = sum_rate

            Q_w = [math.ceil(q["weight"].bit.item())     if hasattr(q["weight"],     "bit") else 16
                   for q in quantizers.values()]
            Q_a = [math.ceil(q["activation"].bit.item()) if hasattr(q["activation"], "bit") else 16
                   for q in quantizers.values()]

            energy_uJ            = analyze_and_compute_energy(
                model, Q_w, Q_a, input_size=(1, 64, 128)
            ) * 1e-6
            result_log["energy"] = energy_uJ
            result_log["EE"]     = sum_rate / (energy_uJ + 1e-8)
            val_loss             = -(sum_rate / fp32_stats["sr"]) \
                                   + current_lambda * (energy_uJ / fp32_stats["en"])
            result_log["val_loss"] = val_loss

            save_result_data(epoch, result_log, file_path)

            if val_loss < best_loss:
                best_loss = val_loss
                best_sr   = sum_rate
                best_en   = energy_uJ
                best_ee   = result_log["EE"]

                save_quantized_model(model, quantizers,
                                     checkpoint_dir / "best_model.pth",
                                     args, epoch, best_loss)
                dump_hw_analysis(
                    model, quantizers,
                    fp32_stats["sr"], fp32_stats["en"], best_sr, best_en,
                    hls_dir / f"hw_analysis_lambda_{current_lambda}.json"
                )
                plot_final_bits(quantizers, plots_dir / f"bits_lambda_{current_lambda}.png")

                logger.info(
                    f"  Ep {epoch:3d} | loss={best_loss:.4f} "
                    f"SR={best_sr:.2f} ({best_sr/fp32_stats['sr']*100:.1f}%) "
                    f"E={best_en:.4f}uJ ({best_en/fp32_stats['en']*100:.1f}%) "
                    f"avg_bits_w={np.mean(Q_w):.1f}"
                )

        global_stats[current_lambda] = {"sr": best_sr, "en": best_en, "ee": best_ee}
        plot_quant_training(file_path, plots_dir, current_lambda,
                            fp32_stats["sr"], fp32_stats["en"])
        logger.info(
            f"  λ={current_lambda} done — "
            f"SR={best_sr:.2f} ({best_sr/fp32_stats['sr']*100:.1f}%), "
            f"E={best_en:.4f}uJ ({best_en/fp32_stats['en']*100:.1f}%)"
        )

    plot_pareto(fp32_stats, global_stats, plots_dir)

    summary = {
        "run_name":   run_name,
        "cfg":        cfg_tag,
        "scenario":   scenario_tag,
        "fp32_stats": fp32_stats,
        "results":    {str(l): v for l, v in global_stats.items()},
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f" Pipeline termine — {run_dir}")

if __name__ == "__main__":
    main()