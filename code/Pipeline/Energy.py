import torch.nn as nn
import math
import torch

def compute_energy_constants(Q):
    # Séparation propre entre Tenseur PyTorch et nombre Python (int/float)
    if isinstance(Q, torch.Tensor):
        Q = torch.clamp(Q, min=1e-5)
    else:
        Q = max(float(Q), 1e-5)
        
    EMAC = (0.857904 * (Q / 16) ** 1.9)
    EM = 2 * EMAC
    EL = EMAC
    return EMAC, EM, EL

# Function to compute total energy (MAINTENANT AVEC Q_W ET Q_A)
def compute_energy(MAC_operations, num_weights, num_activations, Q_W, Q_A):
    EMAC, EM, EL = compute_energy_constants(Q_W)
    _, EM_A, EL_A = compute_energy_constants(Q_A)  # <--- LE 16 EST REMPLACÉ PAR Q_A
    
    if isinstance(Q_W, torch.Tensor):
        sqrt_p_W = torch.sqrt(torch.tensor(64.0) * (Q_W / 16))
        sqrt_p_A = torch.sqrt(torch.tensor(64.0) * (Q_A / 16))
    else:
        sqrt_p_W = math.sqrt(64 * (float(Q_W) / 16))
        sqrt_p_A = math.sqrt(64 * (float(Q_A) / 16))
    
    EC = EMAC * (MAC_operations + 3 * num_activations)
    EW = EM * num_weights + EL * (MAC_operations / sqrt_p_W)
    EA = 2 * EM_A * num_activations + EL_A * (MAC_operations / sqrt_p_A)
    
    EHW = EC + EW + EA
    return EHW

# Function to compute output size
def compute_output_size(layer, input_size):
    if isinstance(layer, nn.Conv2d):
        in_height, in_width = input_size[2], input_size[3]
        out_channels = layer.out_channels
        out_height = (in_height + 2 * layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1) - 1) // layer.stride[0] + 1
        out_width = (in_width + 2 * layer.padding[1] - layer.dilation[1] * (layer.kernel_size[1] - 1) - 1) // layer.stride[1] + 1
        return (input_size[0], out_channels, out_height, out_width)
    elif isinstance(layer, nn.Linear):
        if len(input_size) == 3:
            # Transformer: (batch, seq, in) → (batch, seq, out)
            return (input_size[0], input_size[1], layer.out_features)
        else:
            # MLP: (batch, in) → (batch, out)
            return (input_size[0], layer.out_features)
    elif isinstance(layer, nn.MultiheadAttention):      
        return input_size
    return input_size

def compute_conv2d_macs(layer, input_size):
    in_channels = layer.in_channels
    out_channels = layer.out_channels
    kernel_height, kernel_width = layer.kernel_size
    stride_height, stride_width = layer.stride
    padding_height, padding_width = layer.padding
    dilation_height, dilation_width = layer.dilation
    groups = layer.groups
    in_height = input_size[2]
    in_width = input_size[3]
    out_height = (in_height + 2 * padding_height - dilation_height * (kernel_height - 1) - 1) // stride_height + 1
    out_width = (in_width + 2 * padding_width - dilation_width * (kernel_width - 1) - 1) // stride_width + 1
    macs = in_channels * out_channels * kernel_height * kernel_width * out_height * out_width // groups
    return macs

def compute_linear_macs(layer):
    in_features = layer.in_features
    out_features = layer.out_features
    macs = in_features * out_features
    return macs

def compute_mha_weights(layer: nn.MultiheadAttention) -> int:
    w = layer.in_proj_weight.numel() + layer.out_proj.weight.numel()
    if layer.in_proj_bias is not None:
        w += layer.in_proj_bias.numel() + layer.out_proj.bias.numel()
    return w

def compute_mha_macs(layer: nn.MultiheadAttention, seq_len: int) -> int:
    d = layer.embed_dim          
    return 3*seq_len*d*d + 2*seq_len*seq_len*d + seq_len*d*d

def export_results_to_excel(results, output_path):
    results.to_csv(output_path, index=False)
        
# LA FONCTION PRINCIPALE CORRIGÉE POUR RECEVOIR Q_activations
def analyze_and_compute_energy(model, Q, Q_activations=None, input_size=(1, 2, 4, 64)):
    if isinstance(Q, list):
        if isinstance(Q[0], torch.Tensor):
            Q = list(Q)
            if Q_activations is not None:
                Q_activations = list(Q_activations)
            total_MACs = 0
            total_weights = 0
            total_activations = 0
            As = 0
            total_hardware_energy = 0
            current_size = input_size
            new = -1
            for layer in model.modules():
                if isinstance(layer, nn.Conv2d):
                    new += 1
                    q_w_val = Q[new]
                    q_a_val = Q_activations[new] if (Q_activations is not None and len(Q_activations)>new) else 16
                    total_MACs += compute_conv2d_macs(layer, current_size)
                    total_weights += layer.weight.numel()
                    if layer.bias is not None:
                        total_weights += layer.bias.numel()
                    current_size = compute_output_size(layer, current_size)
                    As += current_size[1] * current_size[2] * current_size[3]
                    total_hardware_energy += compute_energy(total_MACs, total_weights, As, q_w_val, q_a_val)
                    total_MACs = 0
                    total_weights = 0
                    As = 0
                elif isinstance(layer, nn.Linear) and not isinstance(layer, nn.modules.linear.NonDynamicallyQuantizableLinear):
                    new += 1
                    q_w_val = Q[new]
                    q_a_val = Q_activations[new] if (Q_activations is not None and len(Q_activations) > new) else 16
                    total_MACs += compute_linear_macs(layer)
                    total_weights += layer.weight.numel()
                    if layer.bias is not None:
                        total_weights += layer.bias.numel()
                    current_size = compute_output_size(layer, current_size)  # missing update
                    acts = current_size[1] * current_size[2] if len(current_size) == 3 else current_size[1]
                    total_hardware_energy += compute_energy(total_MACs, total_weights, acts, q_w_val, q_a_val)  # once, correct vars
                    total_MACs = 0
                    total_weights = 0
                elif isinstance(layer, nn.MultiheadAttention):
                    new += 1  # <--- CORRECTION : On incrémente UNE SEULE FOIS pour tout le bloc
                    q_w_val = Q[new] if new < len(Q) else 16
                    q_a_val = Q_activations[new] if (Q_activations is not None and len(Q_activations)>new) else 16
                    
                    seq_len = current_size[0]
                    new_proj = ["Q", "K", "V", "O"]
                    for _ in new_proj:
                        macs_proj = compute_mha_macs(layer, seq_len) // 4
                        weights_proj = compute_mha_weights(layer) // 4
                        acts_proj = seq_len * layer.embed_dim
                        total_hardware_energy += compute_energy(
                            macs_proj, weights_proj, acts_proj, q_w_val, q_a_val
                        )
                    current_size = compute_output_size(layer, current_size)
            return total_hardware_energy
        else:
            Q = list(Q)
            if Q_activations is not None:
                Q_activations = list(Q_activations)
            total_hardware_energy = 0.0
            current_size = input_size
            new = -1
            for layer in model.modules():
                if isinstance(layer, nn.Conv2d):
                    new += 1
                    q_w_val = Q[new]
                    q_a_val = Q_activations[new] if (Q_activations is not None and len(Q_activations)>new) else 16
                    macs = compute_conv2d_macs(layer, current_size)
                    wts = layer.weight.numel() + (layer.bias.numel() if layer.bias is not None else 0)
                    current_size = compute_output_size(layer, current_size)
                    acts = current_size[1] * current_size[2] * current_size[3]
                    total_hardware_energy += compute_energy(macs, wts, acts, q_w_val, q_a_val)
                elif (isinstance(layer, nn.Linear) and not isinstance(layer, nn.modules.linear.NonDynamicallyQuantizableLinear)):
                    new += 1
                    q_w_val = Q[new]
                    q_a_val = Q_activations[new] if (Q_activations is not None and len(Q_activations)>new) else 16
                    macs = compute_linear_macs(layer)
                    wts = layer.weight.numel() + (layer.bias.numel() if layer.bias is not None else 0)
                    current_size = compute_output_size(layer, current_size)
                    acts = current_size[1]
                    total_hardware_energy += compute_energy(macs, wts, acts, q_w_val, q_a_val)
                elif isinstance(layer, nn.MultiheadAttention):
                    new += 1  # <--- CORRECTION : On incrémente UNE SEULE FOIS pour tout le bloc
                    q_w_val = Q[new] if new < len(Q) else 16
                    q_a_val = Q_activations[new] if (Q_activations is not None and len(Q_activations)>new) else 16
                    
                    seq_len = current_size[0]
                    d_model = current_size[1]
                    macs_blk = compute_mha_macs(layer, seq_len)
                    wts_blk = compute_mha_weights(layer)
                    for _ in range(4):
                        macs_proj = macs_blk // 4
                        wts_proj = wts_blk // 4
                        acts_proj = seq_len * d_model
                        total_hardware_energy += compute_energy(
                            macs_proj, wts_proj, acts_proj, q_w_val, q_a_val
                        )
                    current_size = compute_output_size(layer, current_size)
            return total_hardware_energy