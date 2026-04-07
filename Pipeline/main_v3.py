import argparse
import logging
import sys
from pathlib import Path
import yaml
import Pipeline.process as process
import quan
import util
import torch
from Pipeline.input_args import input_args
from Pipeline.Energy import analyze_and_compute_energy
from Pipeline.Learner import Learner
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from Pipeline.Transformer import StackedTransformer
from Pipeline.utils import utils_

# ==============================================================================
# TRANSFORMER CONFIGURATIONS
# ==============================================================================
TRANSFORMER_CONFIGS = {
    'tiny':   dict(num_layers=4, embedding_dim=64,  num_heads=4, hidden_dim=512,  dropout=0.05),
    'small':  dict(num_layers=4, embedding_dim=128, num_heads=4, hidden_dim=1024, dropout=0.05),
    'medium': dict(num_layers=4, embedding_dim=128, num_heads=4, hidden_dim=2048, dropout=0.05),
    'base':   dict(num_layers=6, embedding_dim=128, num_heads=4, hidden_dim=2048, dropout=0.05),
    'large':  dict(num_layers=6, embedding_dim=256, num_heads=8, hidden_dim=2048, dropout=0.05),
    'xlarge': dict(num_layers=8, embedding_dim=256, num_heads=8, hidden_dim=4096, dropout=0.05)
}

# ==============================================================================
# ARGUMENT PARSING — must run before input_args()
# ==============================================================================
def parse_overrides():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--scenario',        type=str,   default=None)
    parser.add_argument('--transformer_cfg', type=str,   default='base')
    parser.add_argument('--output_dir',      type=str,   default=None)
    parser.add_argument('--fp32_model_path', type=str,   default=None)
    parser.add_argument('--epochs',          type=int,   default=None)
    parser.add_argument('--batch_size',      type=int,   default=None)
    parser.add_argument('--lambda_reg',      type=float, nargs='+', default=None)
    parser.add_argument('--init_mode',       type=str,   default='pretrained')
    parser.add_argument('--lr_model',        type=float, default=None)  # NEW
    parser.add_argument('--quan',    dest='run_phase2', action='store_true')
    parser.add_argument('--no-quan', dest='run_phase2', action='store_false')

    overrides, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return overrides

# ==============================================================================
# VISUALISATION
# ==============================================================================
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def plot_results(json_file_path):
    data = load_json(json_file_path)
    epochs   = [int(e) for e in data.keys()]
    EE       = [data[str(epoch)]["EE"]       for epoch in epochs]
    sum_rate = [data[str(epoch)]["sum_rate"]  for epoch in epochs]
    energy   = [data[str(epoch)]["energy"]    for epoch in epochs]

    save_dir = os.path.dirname(json_file_path)
    fig1, axs1 = plt.subplots(2, 2, figsize=(14, 10))
    axs1[0, 0].plot(epochs, EE,       marker='o', color='b', linewidth=2)
    axs1[0, 0].set_title('Energy Efficiency vs Epoch', fontsize=12, fontweight='bold')
    axs1[0, 0].grid(True, alpha=0.3)
    axs1[0, 1].plot(epochs, sum_rate, marker='s', color='g', linewidth=2)
    axs1[0, 1].set_title('Sum Rate vs Epoch', fontsize=12, fontweight='bold')
    axs1[0, 1].grid(True, alpha=0.3)
    axs1[1, 0].plot(epochs, energy,   marker='^', color='r', linewidth=2)
    axs1[1, 0].set_title('Energy Consumption vs Epoch', fontsize=12, fontweight='bold')
    axs1[1, 0].set_ylabel('Energy (µJ)')
    axs1[1, 0].grid(True, alpha=0.3)
    ax2      = axs1[1, 1]
    ax2_twin = ax2.twinx()
    ax2.plot(epochs, sum_rate, marker='o', color='g', label='Sum Rate', linewidth=2)
    ax2_twin.plot(epochs, energy, marker='s', color='r', label='Energy', linewidth=2)
    ax2.set_title('Sum Rate & Energy vs Epoch', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(os.path.join(save_dir, 'training_results.png'), dpi=300)
    plt.close(fig1)

def plot_pareto_front(fp32_stats, quant_stats_dict, save_dir):
    labels    = ['FP32 Baseline'] + [f'LSQ (λ={l})' for l in quant_stats_dict.keys()]
    sr_values = [fp32_stats['sr']] + [res['sr'] for res in quant_stats_dict.values()]
    en_values = [fp32_stats['en']] + [res['en'] for res in quant_stats_dict.values()]
    ee_values = [fp32_stats['ee']] + [res['ee'] for res in quant_stats_dict.values()]
    colors    = plt.cm.viridis(np.linspace(0, 0.9, len(quant_stats_dict)))

    plt.figure(figsize=(10, 6))
    plt.scatter(en_values[0], sr_values[0], color='black', marker='*', s=400, label='FP32 Baseline')
    plt.annotate('FP32', (en_values[0], sr_values[0]), xytext=(5, 5), textcoords='offset points', fontweight='bold')
    for i, (lam, color) in enumerate(zip(quant_stats_dict.keys(), colors)):
        plt.scatter(en_values[i+1], sr_values[i+1], color=color, marker='o', s=200, label=f'LSQ (λ={lam})')
        plt.annotate(f'λ={lam}', (en_values[i+1], sr_values[i+1]), xytext=(5, -15), textcoords='offset points')
    plt.title('Global Trade-off', fontsize=14, fontweight='bold')
    plt.xlabel('Energy Consumption (µJ)', fontsize=12)
    plt.ylabel('Sum Rate (bps/Hz)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pareto_front_global.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, ee_values, color=['gray'] + list(colors), edgecolor='black', alpha=0.8)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(ee_values)*0.01),
                 f'{yval:.1f}', ha='center', va='bottom', fontweight='bold')
    plt.title('Energy Efficiency Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Energy Efficiency (bps/Hz / µJ)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'efficiency_bar_chart.png'), dpi=300)
    plt.close()

# ==============================================================================
# UTILITIES
# ==============================================================================
def log_bitlengths(quantizers, result):
    for name, q in quantizers.items():
        result["bit"][name] = q['weight'].bit.item()
        if hasattr(q['activation'], 'bit'):
            result["bit_act"][name] = q['activation'].bit.item()
    return result

def save_result_data(epoch, result, file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}
    data[epoch] = {
        "bit":      result["bit"],
        "bit_act":  result.get("bit_act", {}),
        "sum_rate": result["sum_rate"],
        "EE":       result["EE"],
        "energy":   result["energy"],
        "val_loss": result.get("val_loss", None),
    }
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def save_quantized_model(model, quantizers, save_path, args, epoch, best_metric=None):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'quantizers': {name: {
            'bit':   q['weight'].bit.detach().cpu() if hasattr(q['weight'], 'bit') else None,
            'scale': q['weight'].s.detach().cpu()   if hasattr(q['weight'], 's')   else None,
            'imp':   q.get('imp', 1.0)
        } for name, q in quantizers.items()} if quantizers else {},
        'args':         vars(args) if hasattr(args, '__dict__') else dict(args),
        'best_metric':  best_metric,
    }
    torch.save(checkpoint, save_path)

def resolve_path(target_path, base_dir):
    if not target_path:
        return str(base_dir)
    if os.path.isabs(target_path):
        return target_path
    return os.path.join(base_dir, target_path.lstrip('/'))


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    overrides    = parse_overrides()
    scenario_tag = overrides.scenario
    cfg_tag      = overrides.transformer_cfg
    init_mode    = overrides.init_mode
    tcfg         = TRANSFORMER_CONFIGS[cfg_tag]

    script_dir  = Path(__file__).parent.parent
    CONFIG_PATH = script_dir / "configs" / "base_config.yaml"
    args        = util.get_config(str(CONFIG_PATH))

    if overrides.lambda_reg      is not None: args.lambda_reg      = overrides.lambda_reg
    if overrides.output_dir      is not None: args.output_dir      = overrides.output_dir
    if overrides.fp32_model_path is not None: args.fp32_model_path = overrides.fp32_model_path
    if overrides.epochs          is not None: args.epochs          = overrides.epochs
    run_phase2 = overrides.run_phase2 if overrides.run_phase2 is not None else bool(args.quan)

    output_dir = Path(resolve_path(args.output_dir, script_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    fp32_path         = resolve_path(args.fp32_model_path, script_dir)
    fp32_dir_for_info = Path(fp32_path).parent
    if fp32_dir_for_info.name == "checkpoints":
        fp32_dir_for_info = fp32_dir_for_info.parent
    fp32_info_path = str(fp32_dir_for_info / f"baseline_info_{scenario_tag}_{cfg_tag}.json")

    args.name = f"BitAdapt_{cfg_tag}_{scenario_tag}_{init_mode}"
    log_dir   = util.init_logger(args.name, output_dir, script_dir / "logging.conf")
    logger    = logging.getLogger()

    with open(log_dir / "args.yaml", "w") as yaml_file:
        yaml.safe_dump(dict(args), yaml_file)

    input_args_        = input_args()
    arguments          = input_args_.args
    arguments.scenario = [scenario_tag]

    if args.device.type == "cpu" or not torch.cuda.is_available() or args.device.gpu == []:
        args.device.gpu  = []
        args.device.type = torch.device("cpu")
    else:
        args.device.gpu  = [0]
        args.device.type = torch.device("cuda:0")
        torch.cuda.set_device(0)

    Func = utils_(arguments)
    train_loader, test_loader = Func.Data_Load()
    val_loader = test_loader
    Learner_   = Learner(arguments)

    n_epochs = args.epochs if hasattr(args, 'epochs') and args.epochs else 100

    sample_ch, sample_norm = next(iter(train_loader))
    logger.info(f"Input shape: {sample_norm.shape}")

    # ------------------------------------------------------------------
    # LR policy
    # ------------------------------------------------------------------
    base_batch = 1000
    lr_scale   = min(arguments.batch_size / base_batch, 2.0)
    lr_model   = 2e-4 if cfg_tag in ('large', 'xlarge') else (5e-4 if cfg_tag == 'base' else 1e-3)
    if overrides.lr_model is not None:
        lr_model = overrides.lr_model
        logger.info(f"lr_model overridden → {lr_model:.2e}")

    # lr_bit — override CLI ou fallback yaml
    lr_bit = overrides.lr_bit if overrides.lr_bit is not None else args.lr_bit
    logger.info(f"lr_bit = {lr_bit:.2e}")

    # ==================================================================
    # PHASE 1: FP32 pretraining
    # ==================================================================
    if not run_phase2:

        # Guard — skip si poids existent ET JSON existe
        if os.path.exists(fp32_path) and os.path.exists(fp32_info_path):
            logger.info(f"FP32 model + info JSON already exist, skipping Phase 1.")
            return

        # Guard — poids existent mais JSON manquant → recalcule JSON seulement
        if os.path.exists(fp32_path) and not os.path.exists(fp32_info_path):
            logger.info("FP32 weights exist but info JSON missing — recomputing JSON...")
            model = StackedTransformer(**tcfg).to(args.device.type)
            state_dict = torch.load(fp32_path, map_location=args.device.type)
            model.load_state_dict(state_dict)
            sum_rate, _, _ = process.validate(
                val_loader, model, Learner_.criterium_FDP,
                1, [], args, arguments
            )
            dummy_Q_8 = [8] * sum(
                1 for m in model.modules()
                if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear,
                                  torch.nn.MultiheadAttention))
            )
            en_8bit = analyze_and_compute_energy(
                model, Q=dummy_Q_8, Q_activations=dummy_Q_8,
                input_size=(1, 64, 128)
            ) * 1e-6
            with open(fp32_info_path, 'w') as f:
                json.dump({
                    "sr": sum_rate, "en": en_8bit,
                    "scenario": scenario_tag, "transformer_cfg": cfg_tag,
                    "epochs_trained": "recomputed", "lr_model": lr_model * lr_scale
                }, f, indent=4)
            logger.info(f"✓ JSON recalcule — SR={sum_rate:.4f}, E8={en_8bit:.4f}µJ")
            return

        logger.info("=" * 80)
        logger.info(f" PHASE 1: FP32 PRETRAINING — {cfg_tag} / {scenario_tag}")
        logger.info(f" LR = {lr_model * lr_scale:.2e}  (base={lr_model:.1e} × scale={lr_scale:.2f})")
        logger.info(f" Epochs = {n_epochs}")
        logger.info("=" * 80)

        model = StackedTransformer(**tcfg).to(args.device.type)
        optimizer_fp32 = torch.optim.AdamW(
            model.parameters(), lr=lr_model * lr_scale, weight_decay=0.001
        )
        scheduler_fp32 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_fp32, T_max=n_epochs, eta_min=1e-5
        )

        ckpt_dir         = Path(os.path.dirname(fp32_path)) / "checkpoints"
        save_at          = set(overrides.save_checkpoints) if overrides.save_checkpoints else set()
        save_checkpoints = (n_epochs >= 100)
        if save_checkpoints or save_at:
            ckpt_dir.mkdir(parents=True, exist_ok=True)

        best_sum_rate = -float('inf')

        def criterion_fp32_wrapper(FDP, channel, q=None, m=None, **kwargs):
            return Learner_.criterium_FDP(FDP, channel)

        for epoch in range(1, n_epochs + 1):
            loss, _, _     = process.train(
                train_loader, model, criterion_fp32_wrapper,
                optimizer_fp32, None, epoch, [], args, arguments, None
            )
            sum_rate, _, _ = process.validate(
                val_loader, model, Learner_.criterium_FDP,
                epoch, [], args, arguments
            )
            scheduler_fp32.step()
            logger.info(
                f"FP32 Epoch {epoch:3d}/{n_epochs} | "
                f"Loss: {loss:.4f} | SR: {sum_rate:.4f} | "
                f"LR: {scheduler_fp32.get_last_lr()[0]:.2e}"
            )

            if sum_rate > best_sum_rate:
                best_sum_rate = sum_rate
                os.makedirs(os.path.dirname(fp32_path), exist_ok=True)
                torch.save(model.state_dict(), fp32_path)

                # ← CORRECTION : référence 8-bit, pas FP32
                dummy_Q_8 = [8] * sum(
                    1 for m in model.modules()
                    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear,
                                      torch.nn.MultiheadAttention))
                )
                en_8bit = analyze_and_compute_energy(
                    model, Q=dummy_Q_8, Q_activations=dummy_Q_8,
                    input_size=(1, 64, 128)
                ) * 1e-6

                with open(fp32_info_path, 'w') as f:
                    json.dump({
                        "sr":              best_sum_rate,
                        "en":              en_8bit,  # ← 8-bit ref
                        "scenario":        scenario_tag,
                        "transformer_cfg": cfg_tag,
                        "epochs_trained":  epoch,
                        "lr_model":        lr_model * lr_scale
                    }, f, indent=4)
                logger.info(f"✓ New best SR={best_sum_rate:.4f}, E8={en_8bit:.4f}µJ")

            # Checkpoints
            if epoch in save_at:
                ckpt_path = ckpt_dir / f"fp32_epoch_{epoch}.pth"
                torch.save(model.state_dict(), ckpt_path)
                logger.info(f"Checkpoint sauvegarde : fp32_epoch_{epoch}.pth")
            elif save_checkpoints and epoch % 20 == 0:
                ckpt_path = ckpt_dir / f"fp32_epoch_{epoch}.pth"
                torch.save(model.state_dict(), ckpt_path)
                logger.info(f"Checkpoint ep{epoch} (every-20)")

        logger.info(f"Phase 1 complete. Best SR: {best_sum_rate:.4f}")
        return

    # ==================================================================
    # PHASE 2: Quantization sweep
    # ==================================================================
    logger.info("=" * 80)
    logger.info(f" PHASE 2: QUANTIZATION — {cfg_tag} / {scenario_tag} / {init_mode}")
    logger.info("=" * 80)

    if not os.path.exists(fp32_info_path):
        logger.error(f"FP32 info JSON not found: {fp32_info_path}")
        return

    with open(fp32_info_path, 'r') as f:
        info = json.load(f)
    fp32_stats = {
        'sr': info["sr"],
        'en': info["en"],
        'ee': info["sr"] / (info["en"] + 1e-8)
    }
    logger.info(f"8-bit ref — SR={fp32_stats['sr']:.2f}, E={fp32_stats['en']:.4f}µJ, "
                f"EE={fp32_stats['ee']:.2f}")

    if init_mode == 'pretrained' and not os.path.exists(fp32_path):
        logger.error(f"FP32 weights not found: {fp32_path}")
        return

    lambdas_to_test       = args.lambda_reg if isinstance(args.lambda_reg, list) else [args.lambda_reg]
    global_tradeoff_stats = {}

    for current_lambda in lambdas_to_test:
        logger.info(f"\n{'='*60}\n LAMBDA = {current_lambda}\n{'='*60}")

        model = StackedTransformer(**tcfg).to(args.device.type)
        if init_mode == 'pretrained':
            state_dict = torch.load(fp32_path, map_location=args.device.type)
            model.load_state_dict(state_dict)
            logger.info(f"✓ Loaded FP32 weights from {fp32_path}")
        else:
            logger.info("Scratch mode — random init")

        Learner_ = Learner(arguments,
                           fp32_energy_ref=fp32_stats['en'],
                           fp32_sr_ref=fp32_stats['sr'])

        replaced_modules, quantizers = quan.find_modules_to_quantize(model, args.quan)
        model      = quan.replace_module_by_names(model, replaced_modules)
        quantizers = quan.fix_quantizer_references(model, quantizers)
        quantizers = quan.calibrate_activation_quantizers(
            model, quantizers, train_loader, args.device.type, num_batches=10
        )

        lambda_dir     = log_dir / f"lambda_{current_lambda}"
        lambda_dir.mkdir(exist_ok=True, parents=True)
        file_path      = os.path.join(lambda_dir, f"results_L_{current_lambda}.json")
        checkpoint_dir = lambda_dir / "checkpoints"

        bit_params   = [p for n, p in model.named_parameters() if 'bit' in n.split('.')[-1]]
        other_params = [p for n, p in model.named_parameters() if 'bit' not in n.split('.')[-1]]
        optimizer = torch.optim.AdamW([
            {'params': bit_params,   'lr': lr_bit,              'weight_decay': arguments.wd, 'amsgrad': True},
            {'params': other_params, 'lr': lr_model * lr_scale, 'weight_decay': 0.001,        'amsgrad': True},
        ])

        quant_epochs    = args.epochs if hasattr(args, 'epochs') and args.epochs else 80
        scheduler_quant = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=quant_epochs, eta_min=1e-5  # ← 1e-5 pas 1e-6
        )

        def criterion_quant_wrapper(FDP, channel, q, m, **kwargs):
            return Learner_.criterium_with_bitpruning(
                FDP, channel, q, m, lambda_reg=current_lambda
            )

        result_log = {"bit": {}, "bit_act": {}, "sum_rate": None,
                      "EE": None, "energy": None, "val_loss": None}
        best_loss = float('inf')
        best_sr = best_en = best_ee_at_loss = 0.0

        for epoch in range(1, quant_epochs + 1):
            loss, _, _     = process.train(train_loader, model, criterion_quant_wrapper,
                                           optimizer, None, epoch, [], args, arguments, quantizers)
            sum_rate, _, _ = process.validate(val_loader, model, Learner_.criterium_FDP,
                                              epoch, [], args, arguments)
            scheduler_quant.step()

            result_log = log_bitlengths(quantizers, result_log)
            result_log["sum_rate"] = sum_rate

            Q_w = []
            for q in quantizers.values():
                if hasattr(q['weight'], 'bit'):
                    w_bit = math.ceil(q['weight'].bit.item())
                    if 'weight_out' in q and hasattr(q['weight_out'], 'bit'):
                        w_bit = (w_bit + math.ceil(q['weight_out'].bit.item())) // 2
                else:
                    w_bit = 16
                Q_w.append(w_bit)
            Q_a = [math.ceil(q['activation'].bit.item()) if hasattr(q['activation'], 'bit') else 16
                   for q in quantizers.values()]

            energy_uJ            = analyze_and_compute_energy(
                model, Q_w, Q_a, input_size=(1, 64, 128)
            ) * 1e-6
            result_log["energy"] = energy_uJ
            result_log["EE"]     = sum_rate / (energy_uJ + 1e-8)

            val_loss               = -(sum_rate / fp32_stats['sr']) \
                                     + current_lambda * (energy_uJ / fp32_stats['en'])
            result_log["val_loss"] = val_loss

            save_result_data(epoch, result_log, file_path)

            if val_loss < best_loss:
                best_loss       = val_loss
                best_sr         = sum_rate
                best_en         = energy_uJ
                best_ee_at_loss = result_log["EE"]
                save_quantized_model(model, quantizers,
                                     checkpoint_dir / "best_model.pth",
                                     args, epoch, best_loss)
                logger.info(f"✓ Ep {epoch} | Loss={best_loss:.4f} "
                            f"SR={best_sr:.2f} ({best_sr/fp32_stats['sr']*100:.1f}%) "
                            f"E={best_en:.4f}µJ ({best_en/fp32_stats['en']*100:.1f}%) "
                            f"EE={best_ee_at_loss:.2f} "
                            f"LR={scheduler_quant.get_last_lr()[0]:.2e}")

        global_tradeoff_stats[current_lambda] = {
            'sr': best_sr, 'en': best_en, 'ee': best_ee_at_loss
        }
        plot_results(file_path)

    plot_pareto_front(fp32_stats, global_tradeoff_stats, log_dir)
    logger.info("Training complete.")

if __name__ == "__main__":
    main()