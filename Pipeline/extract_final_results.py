import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

REAL_FP32_SR = 23.80951570510864
REAL_FP32_ENERGY = 5.293803012095999
REAL_FP32_EE = REAL_FP32_SR / REAL_FP32_ENERGY

NUM_TRANSFORMER_LAYERS = 4

def get_transformer_structure():
    d_model = 128
    d_ff = 1024
    return {
        "embedding": (d_model * d_model) + d_model,
        "attention": (4 * d_model * d_model) + (4 * d_model),
        "ff0":       (d_model * (d_ff * 2)) + (d_ff * 2),
        "ff3":       (d_ff * d_model) + d_model,
        "output":    (d_model * d_model) + d_model,
    }

# FIX: safe string matching + correct total_params with num_layers
def calculate_model_size_mb(bit_dict, structure, num_layers=NUM_TRANSFORMER_LAYERS):
    total_bits = 0
    for name, bits in bit_dict.items():
        clean = name.replace("module.", "")
        if clean == "embedding":
            param_count = structure["embedding"]
        elif clean == "output":
            param_count = structure["output"]
        elif "attention" in clean:
            param_count = structure["attention"]
        elif "feed_forward.0" in clean:
            param_count = structure["ff0"]
        elif "feed_forward.3" in clean:
            param_count = structure["ff3"]
        else:
            param_count = 0
        total_bits += param_count * np.ceil(bits)

    total_params = (
        structure["embedding"] +
        structure["output"] +
        num_layers * (structure["attention"] + structure["ff0"] + structure["ff3"])
    )
    return total_bits / (8 * 1024 * 1024), total_params

def get_fp32_size_mb(structure, num_layers=NUM_TRANSFORMER_LAYERS):
    # FIX: was sum(structure.values()) which missed the x4 for num_layers
    total_params = (
        structure["embedding"] +
        structure["output"] +
        num_layers * (structure["attention"] + structure["ff0"] + structure["ff3"])
    )
    return (total_params * 32) / (8 * 1024 * 1024)

def extract_run_data(base_run_dir):
    structure = get_transformer_structure()
    json_files = glob.glob(os.path.join(base_run_dir, "lambda_*", "results_L_*.json"))

    results, layer_bits_cont, layer_bits_disc = [], {}, {}
    layer_act_cont, layer_act_disc = {}, {}

    for file_path in json_files:
        lam = float(os.path.basename(file_path).split('_L_')[1].split('.json')[0])
        with open(file_path, 'r') as f:
            data = json.load(f)

        # FIX: use val_loss (min) not EE (max)
        best_epoch = min(data.keys(), key=lambda k: data[k]["val_loss"])
        best_data = data[best_epoch]

        size_mb, _ = calculate_model_size_mb(best_data["bit"], structure)
        results.append({
            "Lambda": lam, "Size_MB": size_mb,
            "SumRate": best_data["sum_rate"],
            "Energy":  best_data["energy"],
            "EE":      best_data["EE"]
        })

        layer_names = [
            k.replace("module.", "").replace("layers.", "L").replace("feed_forward", "FF")
            for k in best_data["bit"].keys()
        ]
        layer_bits_cont[lam] = list(best_data["bit"].values())
        layer_bits_disc[lam] = [np.ceil(v) for v in best_data["bit"].values()]

        if "bit_act" in best_data and best_data["bit_act"]:
            layer_act_cont[lam] = list(best_data["bit_act"].values())
            layer_act_disc[lam] = [np.ceil(v) for v in best_data["bit_act"].values()]

    fp32_size = get_fp32_size_mb(structure)
    return (
        pd.DataFrame(results).sort_values("Lambda").reset_index(drop=True),
        layer_bits_cont, layer_bits_disc,
        layer_act_cont, layer_act_disc,
        layer_names, fp32_size
    )

# --- Everything below is unchanged ---
def plot_combined_analysis(dir_pre, dir_fs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df_pre, hm_cont_pre, hm_disc_pre, act_cont_pre, act_disc_pre, layers, fp32_size = extract_run_data(dir_pre)
    df_fs,  hm_cont_fs,  hm_disc_fs,  act_cont_fs,  act_disc_fs,  _,      _        = extract_run_data(dir_fs)

    df = pd.merge(df_pre, df_fs, on="Lambda", suffixes=('_Pre', '_FS'), how='outer').sort_values('Lambda')
    lambdas_str = [str(l) for l in df['Lambda']]
    x = np.arange(len(lambdas_str))
    width = 0.35
    color_qat  = '#1E88E5'
    color_rand = '#D81B60'

    # Pareto
    plt.figure(figsize=(10, 6))
    plt.plot(df['Energy_Pre'], df['SumRate_Pre'], marker='o', markersize=9, linewidth=2.5, color=color_qat,  label='QAT (FP32 Init.)')
    plt.plot(df['Energy_FS'],  df['SumRate_FS'],  marker='^', markersize=9, linewidth=2.5, color=color_rand, label='Random Init.')
    for i, row in df.iterrows():
        offset_qat = (0, 10)
        offset_fs  = (0, -18)
        if row['Lambda'] == 5.0:  offset_qat = (-15, 12)
        if row['Lambda'] == 10.0: offset_fs  = (15, -20)
        if pd.notna(row['Energy_Pre']):
            plt.annotate(f"$\\lambda$={row['Lambda']}", (row['Energy_Pre'], row['SumRate_Pre']), textcoords="offset points", xytext=offset_qat, ha='center', fontsize=9, color=color_qat)
        if pd.notna(row['Energy_FS']):
            plt.annotate(f"$\\lambda$={row['Lambda']}", (row['Energy_FS'],  row['SumRate_FS']),  textcoords="offset points", xytext=offset_fs,  ha='center', fontsize=9, color=color_rand)
    plt.axhline(y=REAL_FP32_SR, color='gray', linestyle='--', linewidth=2, label='FP32 Reference')
    plt.title('Ultra-Low Energy Regime', fontsize=14, fontweight='bold')
    plt.xlabel('Energy Consumption (µJ)', fontsize=12)
    plt.ylabel('Sum Rate (bps/Hz)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, max(df['Energy_Pre'].max(), df['Energy_FS'].max()) * 1.15)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pareto_zoomed.png"), dpi=300)
    plt.close()

    # Sum Rate vs Lambda
    plt.figure(figsize=(10, 5))
    plt.axhline(y=REAL_FP32_SR, color='gray', linestyle='--', linewidth=2, label='FP32 Reference')
    plt.plot(lambdas_str, df['SumRate_Pre'], marker='o', color=color_qat,  linewidth=2.5, label='QAT (FP32 Init.)')
    plt.plot(lambdas_str, df['SumRate_FS'],  marker='^', color=color_rand, linewidth=2.5, label='Random Init.')
    plt.title('Sum Rate across Penalties (λ)', fontsize=14, fontweight='bold')
    plt.xlabel('Penalty Lambda (λ)', fontsize=12)
    plt.ylabel('Sum Rate (bps/Hz)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sumrate_vs_lambda.png"), dpi=300)
    plt.close()

    # EE bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axhline(y=REAL_FP32_EE, color='gray', linestyle='--', linewidth=2, label=f'FP32 Baseline ({REAL_FP32_EE:.1f})')
    rects1 = ax.bar(x - width/2, df['EE_Pre'], width, label='QAT (FP32 Init.)', color=color_qat,  edgecolor='black', alpha=0.9)
    rects2 = ax.bar(x + width/2, df['EE_FS'],  width, label='Random Init.',     color=color_rand, edgecolor='black', alpha=0.9)
    for rects in [rects1, rects2]:
        for rect in rects:
            h = rect.get_height()
            if not np.isnan(h):
                ax.annotate(f'{h:.1f}', xy=(rect.get_x() + rect.get_width()/2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title('Energy Efficiency Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(lambdas_str)
    ax.set_ylabel('Efficiency (bps/Hz / µJ)')
    ax.set_ylim(0, max(df['EE_Pre'].max(), df['EE_FS'].max()) * 1.15)
    ax.legend(loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ee_combined.png"), dpi=300)
    plt.close()

    # Compression ratio
    df['Comp_Pre'] = fp32_size / df['Size_MB_Pre']
    df['Comp_FS']  = fp32_size / df['Size_MB_FS']
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, df['Comp_Pre'], width, label='QAT (FP32 Init.)', color=color_qat,  edgecolor='black')
    ax.bar(x + width/2, df['Comp_FS'],  width, label='Random Init.',     color=color_rand, edgecolor='black')
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=2, label='FP32 Baseline (1x)')
    ax.set_title('Model Compression Ratio (Relative to FP32)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(lambdas_str)
    ax.set_ylabel('Compression Factor')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "compression_ratio.png"), dpi=300)
    plt.close()

    # Heatmaps
    def plot_dual_heatmap(data_dict_pre, data_dict_fs, title, filename, vmin, vmax, format_type='%.1f'):
        sorted_lambdas = sorted([l for l in df['Lambda'] if l in data_dict_pre and l in data_dict_fs])
        if not sorted_lambdas:
            return
        data_pre = np.array([data_dict_pre[l] for l in sorted_lambdas])
        data_fs  = np.array([data_dict_fs[l]  for l in sorted_lambdas])
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        for ax, data, title_sub in zip(axes, [data_pre, data_fs], ["QAT (FP32 Init.)", "Random Init."]):
            im = ax.imshow(data, cmap='YlGnBu', aspect='auto', vmin=vmin, vmax=vmax)
            ax.set_title(title_sub, fontweight='bold')
            ax.set_xticks(range(len(layers)))
            ax.set_xticklabels(layers, rotation=45, ha='right')
            ax.set_yticks(range(len(sorted_lambdas)))
            ax.set_yticklabels(sorted_lambdas)
            ax.set_ylabel("Penalty (λ)")
            for i in range(len(sorted_lambdas)):
                for j in range(len(layers)):
                    ax.text(j, i, format_type % data[i, j], ha="center", va="center",
                            color="white" if data[i, j] > (vmax / 2) else "black", fontsize=8)
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), pad=0.02)
        cbar.set_label('Bit-width', rotation=270, labelpad=15)
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    plot_dual_heatmap(hm_cont_pre, hm_cont_fs, "WEIGHTS: Mathematical Intent",  "heatmap_weights_continuous.png", 1, 8,  '%.1f')
    plot_dual_heatmap(hm_disc_pre, hm_disc_fs, "WEIGHTS: Hardware Reality",     "heatmap_weights_discrete.png",  1, 8,  '%d')
    if act_cont_pre and act_cont_fs:
        plot_dual_heatmap(act_cont_pre, act_cont_fs, "ACTIVATIONS: Mathematical Intent", "heatmap_activations_continuous.png", 1, 12, '%.1f')
        plot_dual_heatmap(act_disc_pre, act_disc_fs, "ACTIVATIONS: Hardware Reality",    "heatmap_activations_discrete.png",   1, 12, '%d')

    # LaTeX table
    print("\n" + "="*50 + "\nLATEX TABLE\n" + "="*50)
    df_latex = df[['Lambda', 'SumRate_Pre', 'Energy_Pre', 'EE_Pre', 'SumRate_FS', 'Energy_FS', 'EE_FS']]
    print(df_latex.to_latex(index=False, float_format="{:.3f}".format))

if __name__ == "__main__":
    dir_pretrained    = "/export/tmp/sala/bitadapt_result/BitAdapt_Transformer_pretrained_20260222-222741"
    dir_from_scratch  = "/export/tmp/sala/bitadapt_result/BitAdapt_Transformer_from_scratch_20260222-213616"
    output_combined_dir = "/users/sala/Documents/ELE6310/BitAdapt/results"
    plot_combined_analysis(dir_pretrained, dir_from_scratch, output_combined_dir)