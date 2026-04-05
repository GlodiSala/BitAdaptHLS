import torch
import math
import os

MODEL_PATH = "/users/sala/Documents/ELE6310/BitAdapt/results/BitAdapt_Transformer_20260219-030416/checkpoints/best_model.pth"

EPOCH_70_DATA = {
    "bit": {
        "module.embedding": 2.016559600830078,
        "module.layers.0.attention": 2.0,
        "module.layers.0.feed_forward.0": 2.0,
        "module.layers.0.feed_forward.3": 2.0,
        "module.layers.1.attention": 2.0,
        "module.layers.1.feed_forward.0": 2.0,
        "module.layers.1.feed_forward.3": 2.0,
        "module.layers.2.attention": 2.0,
        "module.layers.2.feed_forward.0": 2.0,
        "module.layers.2.feed_forward.3": 2.0,
        "module.layers.3.attention": 2.0,
        "module.layers.3.feed_forward.0": 2.0,
        "module.layers.3.feed_forward.3": 2.0,
        "module.output": 5.234121322631836
    }
}

# FP32 reference — hardcoded from architecture, not from model_state
D_MODEL, D_FF, N_LAYERS = 128, 1024, 4
FP32_PARAMS = (
    (D_MODEL * D_MODEL + D_MODEL) +        # embedding
    (D_MODEL * D_MODEL + D_MODEL) +        # output
    N_LAYERS * (
        (4 * D_MODEL * D_MODEL + 4 * D_MODEL) +   # attention
        (D_MODEL * D_FF * 2 + D_FF * 2) +          # ff0
        (D_FF * D_MODEL + D_MODEL)                  # ff3
    )
)

def get_quantizer_key(tensor_name, bit_dict):
    """Walk up the name hierarchy to find the matching bit_dict key."""
    base = tensor_name.replace('.weight', '').replace('.bias', '')
    if base in bit_dict:
        return base
    parts = base.split('.')
    for i in range(len(parts), 0, -1):
        candidate = '.'.join(parts[:i])
        if candidate in bit_dict:
            return candidate
    return None

def main():
    print("=" * 60)
    print("  CALCUL DE LA TAILLE MATÉRIELLE FPGA (MIMO TRANSFORMER)")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"[!] Fichier modèle introuvable : {MODEL_PATH}")
        return

    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        model_state = checkpoint['state_dict']
    elif 'model' in checkpoint:
        model_state = checkpoint['model']
    else:
        model_state = checkpoint

    bit_dict = EPOCH_70_DATA["bit"]

    total_params_hw = 0
    total_bits_hardware = 0

    print("\n" + "-" * 95)
    print(f"{'Tensor':<45} | {'Params':>8} | {'Bit Learned':>11} | {'Bit HW':>6} | {'Size HW (Ko)':>12}")
    print("-" * 95)

    for name, tensor in model_state.items():
        # Only count actual model parameters, skip quantizer internals (s, bit)
        if 'weight' not in name and 'bias' not in name:
            continue
        # Skip LSQ internal parameters stored in state_dict
        if any(skip in name for skip in ['quan_w_fn', 'quan_a_fn', 'quan_w_in_fn', 'quan_w_out_fn']):
            continue

        num_params = tensor.numel()
        key = get_quantizer_key(name, bit_dict)
        bit_learned = bit_dict[key] if key else 16.0
        bit_hardware = math.ceil(bit_learned)

        total_params_hw += num_params
        total_bits_hardware += num_params * bit_hardware

        size_hw_kb = (num_params * bit_hardware) / (8 * 1024)
        print(f"{name:<45} | {num_params:>8} | {bit_learned:>11.4f} | {bit_hardware:>6} | {size_hw_kb:>12.2f}")

    size_float32_kb  = (FP32_PARAMS * 32) / (8 * 1024)
    size_hardware_kb = total_bits_hardware / (8 * 1024)
    compression_ratio = size_float32_kb / size_hardware_kb

    print("\n" + "=" * 60)
    print(f"  RÉSUMÉ (EPOCH 70)")
    print("=" * 60)
    print(f"Paramètres FP32 (architecture) : {FP32_PARAMS:,}")
    print(f"Taille FP32                    : {size_float32_kb:,.2f} Ko")
    print(f"Taille compressée (HW)         : {size_hardware_kb:,.2f} Ko")
    print(f"Facteur de compression         : {compression_ratio:.2f}x")
    print("=" * 60)

if __name__ == "__main__":
    main()