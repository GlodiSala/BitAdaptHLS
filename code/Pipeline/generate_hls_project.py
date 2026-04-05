"""
generate_hls_project.py - Generates complete HLS Transformer project with Delayed Scaling.
Delayed Scaling: weights stored as ap_int<b_w> integers + float scales per-channel.
MAC: acc_int = sum(W_int * x_int), output = acc_int * S_w[i] * s_act + B[i]
LUT vs DSP: fabric (LUT) when b_w + b_a <= 10, dsp otherwise.
Latency: inner MAC loops pipelined with II=1.

Usage:
  python Pipeline/generate_hls_project.py --checkpoint /export/tmp/sala/results_fpga/FPGA_small_fpga_stecath/phase2_quant/lambda_5.0/checkpoints/best_model.pth --hw_analysis /export/tmp/sala/results_fpga/FPGA_small_fpga_stecath/hls_analysis/hw_analysis_lambda_5.0.json --output_dir ./hls_ds_small_fpga_l5  --scenario stecath --seed 11
"""

import argparse, json, math, re, sys
import numpy as np
import torch
from pathlib import Path


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--hw_analysis",  required=True)
    p.add_argument("--output_dir",   default="./hls_output")
    p.add_argument("--project_root", default="/users/sala/Documents/ELE6310/ELE6310E")
    p.add_argument("--scenario",     default="stecath")
    p.add_argument("--seed",         type=int, default=11)
    cli, _ = p.parse_known_args()
    return cli


# ==============================================================================
# Data loader
# ==============================================================================

def get_data_loader(project_root, scenario, seed=11):
    sys.path.insert(0, project_root)
    from Pipeline.input_args import input_args
    from Pipeline.utils import utils_

    _saved = sys.argv[:]
    sys.argv = [sys.argv[0], "--scenario", scenario]
    input_args_ = input_args()
    arguments   = input_args_.args
    arguments.scenario = [scenario]
    sys.argv = _saved

    _, loader = utils_(arguments).Data_Load()

    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=True,
        num_workers=getattr(loader, 'num_workers', 0),
        generator=g,
        worker_init_fn=lambda wid: np.random.seed(seed + wid),
    )


def load_sample(project_root, scenario, seed):
    loader   = get_data_loader(project_root, scenario, seed)
    torch.manual_seed(seed)          # ← AJOUTE ÇA
    np.random.seed(seed)             # ← ET ÇA
    ch, ch_norm = next(iter(loader))
    inputs   = ch_norm[:1].float()
    size     = inputs.size()
    xb       = inputs.permute(0,2,3,1).contiguous().view(size[0], size[2], size[3]*size[1])
    token_np = xb[0].numpy()
    print(f"  Sample: token={token_np.shape} min/max={token_np.min():.4f}/{token_np.max():.4f}")
    return inputs, token_np


# ==============================================================================
# Architecture detection
# ==============================================================================

def detect_architecture(sd, project_root, scenario, seed=11):
    sys.path.insert(0, project_root)
    from main_FPGA import TRANSFORMER_CONFIGS

    layer_indices = sorted(set(
        int(k.split(".")[1]) for k in sd
        if k.startswith("layers.") and k.split(".")[1].isdigit()
    ))
    num_layers = len(layer_indices)
    emb_dim    = sd["embedding.weight"].shape[0]
    token_dim  = sd["embedding.weight"].shape[1]
    hid_dim    = sd["layers.0.feed_forward.0.weight"].shape[0]
    output_dim = sd["output.weight"].shape[0] if "output.weight" in sd else emb_dim
    ffn_keys   = sorted(set(
        int(k.split(".")[3]) for k in sd
        if k.startswith("layers.0.feed_forward.") and
        k.split(".")[3].isdigit() and "weight" in k
    ))

    num_heads = None
    for cfg_name, cfg_vals in TRANSFORMER_CONFIGS.items():
        if (cfg_vals['embedding_dim'] == emb_dim and
            cfg_vals['hidden_dim']    == hid_dim and
            cfg_vals['num_layers']    == num_layers):
            num_heads = cfg_vals['num_heads']
            print(f"    matched TRANSFORMER_CONFIGS['{cfg_name}'] -> num_heads={num_heads}")
            break
    if num_heads is None:
        num_heads = 4
        print(f"    [WARN] config not found, num_heads=4 (default)")

    loader = get_data_loader(project_root, scenario, seed)
    ch, ch_norm = next(iter(loader))
    seq_len = ch_norm.shape[2]

    cfg = dict(
        num_layers=num_layers, layer_indices=layer_indices,
        emb_dim=emb_dim, token_dim=token_dim, hid_dim=hid_dim,
        output_dim=output_dim, num_heads=num_heads,
        head_dim=emb_dim // num_heads, ffn_keys=ffn_keys,
        seq_len=seq_len, project_root=project_root, scenario=scenario,
    )
    print(f"\n  Architecture:")
    for k, v in cfg.items():
        if k not in ('project_root', 'scenario'):
            print(f"    {k:15s} = {v}")
    return cfg


# ==============================================================================
# LSQ activation parameters
# ==============================================================================

def collect_lsq_act_params(sd, cfg):
    sys.path.insert(0, cfg['project_root'])
    import quan
    from Pipeline.Transformer_FPGA import StackedTransformer

    class DotDict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__

    quan_cfg = DotDict(
        act    = DotDict(mode='lsq', bit=8, per_channel=True,
                         symmetric=False, all_positive=False),
        weight = DotDict(mode='lsq', bit=8, per_channel=True,
                         symmetric=False, all_positive=False),
        excepts= DotDict()
    )
    sd_p = {k: v.reshape(1) if v.shape == torch.Size([]) else v
            for k, v in sd.items()}

    model = StackedTransformer(
        num_layers=cfg['num_layers'], embedding_dim=cfg['emb_dim'],
        num_heads=cfg['num_heads'], hidden_dim=cfg['hid_dim'],
        dropout=0, token_dim=cfg['token_dim'],
    )
    replaced, _ = quan.find_modules_to_quantize(model, quan_cfg)
    model = quan.replace_module_by_names(model, replaced)
    model.load_state_dict(sd_p, strict=False)
    model.eval()

    params = {}
    for name, mod in model.named_modules():
        if hasattr(mod, 'quan_a_fn'):
            qa      = mod.quan_a_fn
            s       = abs(float(qa.s))
            b_r     = max(2, round(float(qa.bit)))

            thd_pos = 2**b_r - 1       # clipping LSQ correct
            b_r_hw  = b_r + 1          # type ap_int pour HLS
            params[name] = dict(
                s=s, inv_s=1.0/s if s > 0 else 1.0,
                bit_r=b_r_hw,           # utilisé pour ap_int<b_r_hw> dans le C++
                thd_neg=-thd_pos,
                thd_pos=thd_pos,        # clipping reste 2^b_r - 1
            )
            print(f"    LSQ {name:45s} s={s:+.6f} bit_r={b_r} thd=+-{thd_pos}")

    return params, model


# ==============================================================================
# Delayed Scaling weight utilities
# ==============================================================================

def get_lsq_weight_scale(sd, layer_name, proj_type=None):
    key = layer_name + (".quan_w_out_fn.s" if proj_type == 'out' else ".quan_w_fn.s")
    if key not in sd: return None
    s = sd[key].float().numpy()
    if s.ndim == 0: s = s.reshape(1, 1)
    elif s.ndim == 1: s = s.reshape(-1, 1)
    return s


def get_lsq_weight_bit(sd, layer_name, proj_type=None):
    key = layer_name + (".quan_w_out_fn.bit" if proj_type == 'out' else ".quan_w_fn.bit")
    if key not in sd: return 8.0
    return float(sd[key].float())


def quantize_weights_int(weight, scale, w_bit):
    """Returns integer weight matrix and positive scale per-channel."""
    b_w = max(1, math.ceil(float(w_bit)))
    thd   = 2**b_w - 1
    s_abs = np.abs(scale).reshape(-1, 1)
    w_int = np.clip(np.round(weight / s_abs), -thd, thd).astype(np.int32)
    return w_int, np.abs(scale).flatten(), b_w


def impl_choice(b_w, b_a):
    """fabric (LUT) for small products, dsp for larger ones."""
    return "fabric" if (b_w + b_a) <= 10 else "dsp"


def acc_width(b_w, b_a, in_dim):
    """Accumulator width: enough to avoid overflow."""
    return max(b_w + b_a + int(np.ceil(np.log2(in_dim + 1))), 16)


# ==============================================================================
# Weight file generation — Delayed Scaling
# ==============================================================================
def gen_weight_files_ds(layer_name, weight, bias, scale, w_bit, a_bit, outdir):
    safe = layer_name.replace(".", "_")
    w_int, s_flat, b_w = quantize_weights_int(weight, scale, w_bit)
    # thd = 2^b_w - 1 (déjà correct dans quantize_weights_int)
    # ap_int<b_w+1> pour couvrir [-thd, +thd] = [-(2^b_w-1), +(2^b_w-1)]
    b_w_hw = b_w + 1           # type HLS — un bit de plus pour le signe
    b_a    = max(2, int(round(float(a_bit))))
    b_a_hw = b_a + 1      # +1 pour le signe, comme b_w_hw
    out_dim, in_dim = weight.shape
    ab    = acc_width(b_w_hw, b_a_hw, in_dim)
    impl  = impl_choice(b_w_hw, b_a_hw)
    storage = "LUTRAM" if b_w_hw <= 4 else "BRAM"

    # .h — types avec +1 bit
    h_lines = [
        f"// Delayed Scaling | {layer_name} | ap_int<{b_w_hw}> x ap_int<{b_a_hw}> -> impl={impl}",
        f"// LSQ thd=+-{2**b_w - 1} (b_w={b_w}), HLS type=ap_int<{b_w_hw}>",
        f"#pragma once",
        f'#include "ap_int.h"',
        f"",
        f"typedef ap_int<{b_w_hw}>    wi_{safe}_t;",
        f"typedef ap_int<{b_a_hw}>    ai_{safe}_t;",
        f"typedef ap_int<{ab}>        acc_{safe}_t;",
        f"",
        f"#define {safe.upper()}_OUT  {out_dim}",
        f"#define {safe.upper()}_IN   {in_dim}",
        f"#define {safe.upper()}_IMPL \"{impl}\"",
        f"",
        f"extern const wi_{safe}_t {safe}_W[{out_dim}][{in_dim}];",
        f"extern const float        {safe}_S[{out_dim}];",
    ]
    if bias is not None:
        h_lines.append(f"extern const float        {safe}_B[{out_dim}];")
    (outdir / f"{safe}.h").write_text("\n".join(h_lines))

    # .cpp
    cpp_lines = [f'#include "{safe}.h"']
    cpp_lines.append(f"const wi_{safe}_t {safe}_W[{out_dim}][{in_dim}] = {{")
    for i in range(out_dim):
        row = ", ".join([str(int(v)) for v in w_int[i]])
        cpp_lines.append(f"  {{{row}}},")
    cpp_lines.append("};")
    s_vals = ", ".join([f"{float(v):.10f}f" for v in s_flat])
    cpp_lines.append(f"const float {safe}_S[{out_dim}] = {{{s_vals}}};")
    if bias is not None:
        b_vals = ", ".join([f"{float(v):.10f}f" for v in bias.flatten()])
        cpp_lines.append(f"const float {safe}_B[{out_dim}] = {{{b_vals}}};")
    (outdir / f"{safe}.cpp").write_text("\n".join(cpp_lines))

    # meta utilise b_w_hw et b_a_hw pour la génération des blocs HLS
    meta = dict(b_w=b_w_hw, b_a=b_a_hw, ab=ab, impl=impl,
                out_dim=out_dim, in_dim=in_dim, has_bias=bias is not None)
    print(f"  ok {safe}: ap_int<{b_w_hw}>xap_int<{b_a_hw}> ab={ab}b impl={impl} storage={storage}")
    return meta

def gen_rmsnorm_files(layer_name, gamma, outdir):
    """RMSNorm gamma stored as float — unchanged from original."""
    safe = layer_name.replace(".", "_")
    g_q  = np.clip(np.round(gamma / 2**(-4)) * 2**(-4),
                   -(2**3 - 2**(-4)), 2**3 - 2**(-4))
    h = "\n".join([
        f"// RMSNorm | {layer_name} | gamma float",
        f"#pragma once",
        f'#include "ap_fixed.h"',
        f"typedef ap_fixed<8,4>   w_{safe}_t;",
        f"typedef ap_fixed<24,12> acc_{safe}_t;",
        f"extern const w_{safe}_t {safe}_gamma[{len(gamma)}];",
    ])
    (outdir / f"{safe}.h").write_text(h)

    vals = ", ".join([f"{v:.10f}" for v in g_q.flatten()])
    (outdir / f"{safe}.cpp").write_text(
        f'#include "{safe}.h"\n'
        f"const w_{safe}_t {safe}_gamma[{len(gamma)}] = {{{vals}}};"
    )
    return safe


# ==============================================================================
# C++ code generation helpers
# ==============================================================================

def mac_loop_ds(safe, out_dim, in_dim, b_w, b_a, ab, impl,
                x_int_var, s_act_var, out_var, has_bias,
                seq_s=None, activation="none", indent="    "):
    """
    Generates the Delayed Scaling MAC loop.
    seq_s: loop variable name for sequence dimension (None if no seq loop here)
    activation: "none" | "relu"
    
    acc_int = sum_j W[i][j] * x_int[j]
    y[i]    = acc_int * S[i] * s_act + B[i]  (+ ReLU if requested)
    
    Latency: inner j-loop pipelined with II=1
    """
    i = indent
    x_idx = f"{x_int_var}[{seq_s}][j]" if seq_s else f"{x_int_var}[j]"
    o_idx = f"{out_var}[{seq_s}][ii]"  if seq_s else f"{out_var}[ii]"
    bias_str = f"{safe}_B[ii]" if has_bias else "0.0f"

    relu = ""
    if activation == "relu":
        relu = f"{i}    acc_f = acc_f > 0.0f ? acc_f : 0.0f;\n"

    return (
        f"{i}for (int ii = 0; ii < {out_dim}; ii++) {{\n"
        f"{i}    ap_int<{ab}> acc_int = 0;\n"
        f"{i}    MAC_{safe}: for (int j = 0; j < {in_dim}; j++) {{\n"
        f"{i}#pragma HLS PIPELINE II=1\n"
        f"{i}#pragma HLS BIND_OP variable=acc_int op=mul impl={impl}\n"
        f"{i}        acc_int += (ap_int<{ab}>){safe}_W[ii][j]"
        f" * (ap_int<{ab}>){x_idx};\n"
        f"{i}    }}\n"
        f"{i}    float acc_f = (float)acc_int * {safe}_S[ii] * {s_act_var}"
        f" + (float){bias_str};\n"
        f"{relu}"
        f"{i}    {o_idx} = acc_f;\n"
        f"{i}}}\n"
    )

def lsq_to_int_code_ds(var_in, var_out, var_scale, p, b_a, dim, seq=None, indent="    "):
    s     = abs(float(p['s']))
    inv_s = 1.0 / s if s > 0 else 1.0
    thd   = int(p['thd_pos'])  # cohérent avec lsq_int_np
    i     = indent

    if seq is None:
        return (
            f"{i}const float {var_scale} = {s:.10f}f;\n"
            f"{i}ap_int<{b_a}> {var_out}[{dim}];\n"
            f"{i}Q_{var_out}: for (int j = 0; j < {dim}; j++) {{\n"
            f"{i}#pragma HLS PIPELINE II=1\n"
            f"{i}    float _v = roundf((float){var_in}[j] * {inv_s:.8f}f);\n"
            f"{i}    _v = _v < {float(-thd):.1f}f ? {float(-thd):.1f}f"
            f" : (_v > {float(thd):.1f}f ? {float(thd):.1f}f : _v);\n"
            f"{i}    {var_out}[j] = (ap_int<{b_a}>)_v;\n"
            f"{i}}}\n"
        )
    else:
        return (
            f"{i}const float {var_scale} = {s:.10f}f;\n"
            f"{i}ap_int<{b_a}> {var_out}[{seq}][{dim}];\n"
            f"{i}for (int si = 0; si < {seq}; si++) {{\n"
            f"{i}    Q_{var_out}: for (int j = 0; j < {dim}; j++) {{\n"
            f"{i}#pragma HLS PIPELINE II=1\n"
            f"{i}        float _v = roundf((float){var_in}[si][j] * {inv_s:.8f}f);\n"
            f"{i}        _v = _v < {float(-thd):.1f}f ? {float(-thd):.1f}f"
            f" : (_v > {float(thd):.1f}f ? {float(thd):.1f}f : _v);\n"
            f"{i}        {var_out}[si][j] = (ap_int<{b_a}>)_v;\n"
            f"{i}    }}\n"
            f"{i}}}\n"
        )
# ==============================================================================
# HLS block generation — Delayed Scaling
# ==============================================================================

def gen_embedding_block(cfg, outdir, inp_tb, inp_ib, lsq_params, weight_meta):
    D    = cfg['emb_dim']
    T    = cfg['token_dim']  # 128 — input dimension
    S    = cfg['seq_len']
    meta = weight_meta['embedding']
    b_w, b_a, ab, impl = meta['b_w'], meta['b_a'], meta['ab'], meta['impl']

    p = lsq_params.get("embedding",
                        dict(s=0.007, inv_s=143.0, bit_r=5, thd_neg=-31, thd_pos=31))

    # Quantifie input de dim T (token_dim=128), pas D (emb_dim=192)
    q_code = lsq_to_int_code_ds("input", "x_int", "s_act_emb", p, b_a, T, seq=S)

    mac_code = (
        f"    for (int s = 0; s < {S}; s++) {{\n"
        + mac_loop_ds(
            "embedding", D, T, b_w, b_a, ab, impl,
            "x_int", "s_act_emb", "output",
            has_bias=meta['has_bias'], seq_s="s", indent="        ")
        + f"    }}\n"
    )

    (outdir / "embedding_layer.cpp").write_text(
f"""// Delayed Scaling | Embedding | S={S} T={T} D={D}
// w=ap_int<{b_w}> a=ap_int<{b_a}> acc={ab}b impl={impl}
// input: ap_fixed<{inp_tb},{inp_ib}> shape [{S}][{T}]
// output: float shape [{S}][{D}]
#include "transformer_top.h"
#include "embedding.h"
#include "ap_int.h"
#include <math.h>
void embedding_layer(input_t input[{S}][TOKEN_DIM], float output[{S}][EMB_DIM]) {{

{q_code}
{mac_code}}}
""")
    print(f"  ok embedding_layer.cpp (DS T={T} D={D} impl={impl})")



def gen_attention_block(i, cfg, lsq, outdir, weight_meta):
    D   = cfg['emb_dim']
    H   = cfg['num_heads']
    HD  = cfg['head_dim']
    S   = cfg['seq_len']
    ai  = f"layers_{i}_attention_in_proj"
    ao  = f"layers_{i}_attention_out_proj"
    m_ai = weight_meta[ai]
    m_ao = weight_meta[ao]
    inv_sqrt_hd = 1.0 / math.sqrt(HD)

    p = lsq.get(f"layers.{i}.attention",
                dict(s=1.0, inv_s=1.0, bit_r=2, thd_neg=-3, thd_pos=3))

    q_code = lsq_to_int_code_ds(
        "input", "x_int", "s_act_attn", p, m_ai['b_a'], D, seq=S)

    # QKV MAC loop
    qkv_mac = (
        f"    for (int s = 0; s < {S}; s++) {{\n"
        + mac_loop_ds(ai, D*3, D, m_ai['b_w'], m_ai['b_a'], m_ai['ab'], m_ai['impl'],
                      "x_int", "s_act_attn", "qkv",
                      has_bias=m_ai['has_bias'], seq_s="s", indent="        ")
        + f"    }}\n"
    )

    # Out proj MAC — ctx is float so we use float multiply directly
    # (ctx is post-softmax: no integer quantization without re-quantizing)
    out_mac = f"""
    // Out projection — ctx is float (post-softmax), use float MAC
    for (int s = 0; s < {S}; s++) {{
        for (int r = 0; r < {D}; r++) {{
            float acc = {"(float)" + ao + "_B[r]" if m_ao['has_bias'] else "0.0f"};
            OUT_PROJ_{i}: for (int c = 0; c < {D}; c++) {{
#pragma HLS PIPELINE II=1
                // W stored as int, dequantize on the fly
                acc += (float){ao}_W[r][c] * {ao}_S[r] * ctx[s][c];
            }}
            output[s][r] = acc;
        }}
    }}"""

    (outdir / f"attention_layer_{i}.cpp").write_text(
f"""// Delayed Scaling | Attention layer {i} | S={S} H={H} HD={HD}
// QKV: w=ap_int<{m_ai['b_w']}> a=ap_int<{m_ai['b_a']}> impl={m_ai['impl']}
// OutProj: float MAC (ctx is post-softmax float)
#include "{ai}.h"
#include "{ao}.h"
#include "transformer_top.h"
#include "ap_int.h"
#include <math.h>

void attention_layer_{i}(float input[{S}][{D}], float output[{S}][{D}]) {{

{q_code}

    float qkv[{S}][{D*3}];
{qkv_mac}

    // Multi-head attention scores + softmax (float — required for numerical stability)
    float ctx[{S}][{D}];
    for (int h = 0; h < {H}; h++) {{
        int qoff = h * {HD};
        int koff = {D} + h * {HD};
        int voff = {D*2} + h * {HD};

        float scores[{S}][{S}];
        for (int qi = 0; qi < {S}; qi++) {{
            SCORE_{i}_h: for (int kj = 0; kj < {S}; kj++) {{
#pragma HLS PIPELINE II=1
                float dot = 0.0f;
                for (int d = 0; d < {HD}; d++)
                    dot += qkv[qi][qoff+d] * qkv[kj][koff+d];
                scores[qi][kj] = dot * {inv_sqrt_hd:.8f}f;
            }}
        }}

        for (int qi = 0; qi < {S}; qi++) {{
            float mx = scores[qi][0];
            for (int kj = 1; kj < {S}; kj++)
                if (scores[qi][kj] > mx) mx = scores[qi][kj];
            float aw[{S}], sum = 0.0f;
            SOFTMAX_{i}: for (int kj = 0; kj < {S}; kj++) {{
#pragma HLS PIPELINE II=1
                aw[kj] = expf(scores[qi][kj] - mx);
                sum += aw[kj];
            }}
            float inv_sum = 1.0f / (sum + 1e-9f);
            CTX_{i}: for (int d = 0; d < {HD}; d++) {{
#pragma HLS PIPELINE II=1
                float c = 0.0f;
                for (int kj = 0; kj < {S}; kj++)
                    c += aw[kj] * inv_sum * qkv[kj][voff+d];
                ctx[qi][h*{HD}+d] = c;
            }}
        }}
    }}
{out_mac}
}}
""")
    print(f"  ok attention_layer_{i}.cpp (DS QKV impl={m_ai['impl']})")


def gen_ffn_block(i, cfg, lsq, outdir, weight_meta):
    D   = cfg['emb_dim']
    H   = cfg['hid_dim']
    S   = cfg['seq_len']
    fk  = cfg['ffn_keys']
    ff0 = f"layers_{i}_feed_forward_{fk[0]}"
    ff1 = f"layers_{i}_feed_forward_{fk[1]}"
    m0  = weight_meta[ff0]
    m1  = weight_meta[ff1]

    p0 = lsq.get(f"layers.{i}.feed_forward.{fk[0]}",
                  dict(s=0.25, inv_s=4.0, bit_r=2, thd_neg=-3, thd_pos=3))
    p1 = lsq.get(f"layers.{i}.feed_forward.{fk[1]}",
                  dict(s=0.5,  inv_s=2.0, bit_r=2, thd_neg=-3, thd_pos=3))

    q0 = lsq_to_int_code_ds("input", "x_int0", "s_act0", p0, m0['b_a'], D, seq=S)
    q1 = lsq_to_int_code_ds("mid",   "x_int1", "s_act1", p1, m1['b_a'], H, seq=S)

    mac0 = (
        f"    for (int s = 0; s < {S}; s++) {{\n"
        + mac_loop_ds(ff0, H, D, m0['b_w'], m0['b_a'], m0['ab'], m0['impl'],
                      "x_int0", "s_act0", "mid",
                      has_bias=m0['has_bias'], seq_s="s",
                      activation="relu", indent="        ")
        + f"    }}\n"
    )
    mac1 = (
        f"    for (int s = 0; s < {S}; s++) {{\n"
        + mac_loop_ds(ff1, D, H, m1['b_w'], m1['b_a'], m1['ab'], m1['impl'],
                      "x_int1", "s_act1", "output",
                      has_bias=m1['has_bias'], seq_s="s", indent="        ")
        + f"    }}\n"
    )

    (outdir / f"ffn_block_{i}.cpp").write_text(
f"""// Delayed Scaling | FFN layer {i} | S={S} D={D} H={H}
// ff0: w=ap_int<{m0['b_w']}> a=ap_int<{m0['b_a']}> impl={m0['impl']}
// ff1: w=ap_int<{m1['b_w']}> a=ap_int<{m1['b_a']}> impl={m1['impl']}
#include "{ff0}.h"
#include "{ff1}.h"
#include "transformer_top.h"
#include "ap_int.h"
#include <math.h>

void ffn_block_{i}(float input[{S}][{D}], float output[{S}][{D}]) {{

{q0}
    float mid[{S}][{H}];
{mac0}
{q1}
{mac1}}}
""")
    print(f"  ok ffn_block_{i}.cpp (DS ff0={m0['impl']} ff1={m1['impl']})")


def gen_rmsnorm_block(i, norm_idx, gamma_safe, cfg, outdir):
    D  = cfg['emb_dim']
    S  = cfg['seq_len']
    gam = f"{gamma_safe}_gamma"

    (outdir / f"rmsnorm_layer_{i}_{norm_idx}.cpp").write_text(
f"""// RMSNorm layer {i} norm{norm_idx} | S={S} D={D} | ms in float
#include "{gamma_safe}.h"
#include "transformer_top.h"
#include "hls_math.h"

void rmsnorm_layer_{i}_{norm_idx}(float input[{S}][{D}], float output[{S}][{D}]) {{
    for (int s = 0; s < {S}; s++) {{
        float ms_f = 0.0f;
        RMS_SUM_{i}_{norm_idx}: for (int j = 0; j < {D}; j++) {{
#pragma HLS PIPELINE II=1
            float v = input[s][j];
            ms_f += v * v;
        }}
        ms_f = ms_f / {float(D)}f + 1e-5f;
        float inv_rms = 1.0f / hls::sqrt(ms_f);
        RMS_SCALE_{i}_{norm_idx}: for (int j = 0; j < {D}; j++) {{
#pragma HLS PIPELINE II=1
            output[s][j] = input[s][j] * inv_rms * (float){gam}[j];
        }}
    }}
}}
""")
    print(f"  ok rmsnorm_layer_{i}_{norm_idx}.cpp")


def gen_output_block(cfg, outdir, weight_meta):
    D    = cfg['emb_dim']
    OUT  = cfg['output_dim']
    S    = cfg['seq_len']
    meta = weight_meta['output']
    b_w, b_a, ab, impl = meta['b_w'], meta['b_a'], meta['ab'], meta['impl']

    # Output LSQ params will be applied before this block in transformer_top
    mac_code = (
        f"    for (int s = 0; s < {S}; s++) {{\n"
        + mac_loop_ds("output", OUT, D, b_w, b_a, ab, impl,
                      "x_int", "s_act_out", "output_arr",
                      has_bias=meta['has_bias'], seq_s="s", indent="        ")
        + f"    }}\n"
    )

    (outdir / "output_layer.cpp").write_text(
f"""// Delayed Scaling | Output layer | S={S} D={D} OUT={OUT}
// w=ap_int<{b_w}> a=ap_int<{b_a}> acc={ab}b impl={impl}
#include "output.h"
#include "transformer_top.h"
#include "ap_int.h"

void output_layer(float input[{S}][{D}],
                  ap_int<{b_a}> x_int[{S}][{D}],
                  float s_act_out,
                  float output_arr[{S}][{OUT}]) {{
{mac_code}}}
""")
    print(f"  ok output_layer.cpp (DS impl={impl})")


# ==============================================================================
# Top-level transformer
# ==============================================================================

def gen_transformer_top(cfg, norm_safes, lsq, outdir, inp_tb, inp_ib, weight_meta):
    D   = cfg['emb_dim']
    T   = cfg['token_dim']  # 128
    OUT = cfg['output_dim']
    NL  = cfg['num_layers']
    S   = cfg['seq_len']
    fk  = cfg['ffn_keys']

    p_out = lsq.get("output",
                     dict(s=0.1118, inv_s=8.945, bit_r=4, thd_neg=-15, thd_pos=15))
    m_out = weight_meta['output']
    b_a_out = m_out['b_a']

    # Header includes
    layer_includes = []
    for i in cfg['layer_indices']:
        ff0 = f"layers_{i}_feed_forward_{fk[0]}"
        layer_includes += [
            f'#include "{ff0}.h"',
            f'#include "layers_{i}_attention_in_proj.h"',
            f'#include "layers_{i}_norm1.h"',
            f'#include "layers_{i}_norm2.h"',
        ]

    (outdir / "transformer_top.h").write_text(
f"""// Delayed Scaling Transformer | S={S} T={T} D={D} NL={NL} OUT={OUT}
// All intermediate activations in float after first dequantization
#pragma once
#include "ap_fixed.h"
#include "ap_int.h"
#include "embedding.h"
#include "output.h"
{chr(10).join(layer_includes)}

#define NUM_LAYERS  {NL}
#define TOKEN_DIM   {T}
#define EMB_DIM     {D}
#define OUTPUT_DIM  {OUT}
#define SEQ_LEN     {S}

typedef ap_fixed<{inp_tb},{inp_ib}>   input_t;
// Note: intermediate buffers use float (post Delayed Scaling dequantization)
""")

    # Function prototypes
    protos = [
        f"void embedding_layer(input_t input[SEQ_LEN][TOKEN_DIM], float output[SEQ_LEN][EMB_DIM]);",
    ]
    for i in cfg['layer_indices']:
        protos += [
            f"void attention_layer_{i}(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]);",
            f"void rmsnorm_layer_{i}_1(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]);",
            f"void ffn_block_{i}(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]);",
            f"void rmsnorm_layer_{i}_2(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]);",
        ]
    protos.append(
        f"void output_layer(float input[SEQ_LEN][EMB_DIM], "
        f"ap_int<{b_a_out}> x_int[SEQ_LEN][EMB_DIM], "
        f"float s_act_out, float output_arr[SEQ_LEN][OUTPUT_DIM]);"
    )

    # Output quantization code
    s_out     = abs(float(p_out['s']))
    inv_s_out = 1.0 / s_out if s_out > 0 else 1.0
    thd_out   = p_out['thd_pos']

    q_out_code = (
        f"    const float s_act_out = {s_out:.10f}f;\n"
        f"    ap_int<{b_a_out}> out_x_int[SEQ_LEN][EMB_DIM];\n"
        f"    for (int si = 0; si < SEQ_LEN; si++) {{\n"
        f"        Q_OUT: for (int j = 0; j < EMB_DIM; j++) {{\n"
        f"#pragma HLS PIPELINE II=1\n"
        f"            float _v = roundf((float)x[si][j] * {inv_s_out:.8f}f);\n"
        f"            _v = _v < {float(-thd_out):.1f}f ? {float(-thd_out):.1f}f"
        f" : (_v > {float(thd_out):.1f}f ? {float(thd_out):.1f}f : _v);\n"
        f"            out_x_int[si][j] = (ap_int<{b_a_out}>)_v;\n"
        f"        }}\n"
        f"    }}\n"
    )

    # Layer bodies
    body = []
    for i in cfg['layer_indices']:
        body.append(f"""
    // Layer {i}
    {{
        float attn_out[SEQ_LEN][EMB_DIM];
        attention_layer_{i}(x, attn_out);

        // Residual + RMSNorm1
        float norm1_in[SEQ_LEN][EMB_DIM];
        for (int si = 0; si < SEQ_LEN; si++)
            for (int j = 0; j < EMB_DIM; j++)
                norm1_in[si][j] = x[si][j] + attn_out[si][j];
        float norm1_out[SEQ_LEN][EMB_DIM];
        rmsnorm_layer_{i}_1(norm1_in, norm1_out);

        // FFN
        float ffn_out[SEQ_LEN][EMB_DIM];
        ffn_block_{i}(norm1_out, ffn_out);

        // Residual + RMSNorm2
        float norm2_in[SEQ_LEN][EMB_DIM];
        for (int si = 0; si < SEQ_LEN; si++)
            for (int j = 0; j < EMB_DIM; j++)
                norm2_in[si][j] = norm1_out[si][j] + ffn_out[si][j];
        rmsnorm_layer_{i}_2(norm2_in, x);
    }}""")

    (outdir / "transformer_top.cpp").write_text(
f"""// Delayed Scaling Transformer Top | S={S} T={T} D={D} NL={NL}
#include "transformer_top.h"
#include "ap_int.h"
#include <math.h>

{chr(10).join(protos)}

void transformer_top(
    input_t input_flat[{S*T}],
    float   output_flat[{S*OUT}])
{{
#pragma HLS INTERFACE ap_none port=input_flat
#pragma HLS INTERFACE ap_none port=output_flat
#pragma HLS INTERFACE ap_ctrl_none port=return

    // Reshape flat input to 2D token array [SEQ_LEN][TOKEN_DIM]
    input_t input_2d[SEQ_LEN][TOKEN_DIM];
    for (int si = 0; si < SEQ_LEN; si++)
        for (int j = 0; j < TOKEN_DIM; j++)
            input_2d[si][j] = input_flat[si * TOKEN_DIM + j];

    // All intermediate buffers in float
    float x[SEQ_LEN][EMB_DIM];

    // Embedding: TOKEN_DIM -> EMB_DIM
    embedding_layer(input_2d, x);

{"".join(body)}

    // Output quantization to integer + Delayed Scaling MAC
{q_out_code}
    float output_2d[SEQ_LEN][OUTPUT_DIM];
    output_layer(x, out_x_int, s_act_out, output_2d);

    for (int si = 0; si < SEQ_LEN; si++)
        for (int j = 0; j < OUTPUT_DIM; j++)
            output_flat[si * OUTPUT_DIM + j] = output_2d[si][j];
}}
""")
    print(f"  ok transformer_top.cpp + .h (DS, S={S} T={T} D={D})")


# ==============================================================================
# Test vectors and testbench — numpy Delayed Scaling reference
# ==============================================================================

def arr_to_c_float(name, arr):
    vals = ", ".join([f"{v:.8f}" for v in arr.flatten()])
    return f"static const float {name}[{len(arr.flatten())}] = {{{vals}}};\n"

def compute_reference_vectors_ds(sd, cfg, lsq_params, weight_meta, outdir, inputs, inp_ib):
    """
    Numpy simulation EXACTE du forward HLS Delayed Scaling.
    Règles cohérentes avec gen_weight_files_ds et collect_lsq_act_params :
      - Poids : thd = 2^weight_meta[key]['b_w'] - 1  (b_w = b_w_orig+1)
      - Activations : thd = lsq_params[name]['thd_pos']  (= 2^round(bit) - 1)
    """
    D   = cfg['emb_dim']
    S   = cfg['seq_len']
    OUT = cfg['output_dim']
    fk  = cfg['ffn_keys']

    #  Primitives internes 

    def lsq_int_np(arr, p):
        """Quantifie arr avec thd depuis lsq_params (= 2^round(bit) - 1)."""
        s     = abs(float(p['s']))
        inv_s = 1.0 / s if s > 0 else 1.0
        thd   = int(p['thd_pos'])
        v = np.clip(np.round(np.array(arr, dtype=np.float64) * inv_s), -thd, thd)
        return v.astype(np.int32), s

    def load_w_int(w_key, s_key, meta_key):
        """Charge poids en entiers avec thd = 2^b_w_hw - 1 (b_w_hw = b_w_orig+1)."""
        w   = sd[w_key].float().numpy()
        s_w = np.abs(sd[s_key].float().numpy()).flatten() if s_key in sd \
              else np.ones(w.shape[0]) * 0.1
        b_w_hw = weight_meta[meta_key]['b_w']   # déjà b_w_orig+1
        thd    = 2**b_w_hw - 1
        W_int  = np.clip(np.round(w / s_w.reshape(-1,1)), -thd, thd).astype(np.int32)
        return W_int, s_w

    def get_bias(key):
        b = sd.get(key)
        return b.float().numpy() if b is not None else None

    def mac_ds_np(W_int, s_w, x_int, s_act, bias):
        acc    = W_int.astype(np.int64) @ x_int.astype(np.int64)
        result = acc.astype(np.float64) * s_w * float(s_act)
        if bias is not None:
            result += bias.astype(np.float64)
        return result

    def out_proj_float_np(W_int, s_w, ctx, bias):
        """Out proj: ctx est float (post-softmax) → float MAC."""
        w_dq   = W_int.astype(np.float64) * s_w.reshape(-1, 1)
        result = w_dq @ ctx.astype(np.float64)
        if bias is not None:
            result += bias.astype(np.float64)
        return result

    def rmsnorm_np(x, gamma):
        out = np.zeros_like(x, dtype=np.float64)
        for s in range(x.shape[0]):
            ms     = float(np.mean(x[s]**2)) + 1e-5
            out[s] = x[s] / np.sqrt(ms) * gamma.astype(np.float64)
        return out

    def read_gamma(layer_key):
        g  = sd[layer_key].float().numpy()
        q  = 2**(-4); mv = 2**3 - q
        return np.clip(np.round(g / q) * q, -mv, mv)

    #  Input quantification ap_fixed<16,inp_ib> 
    size       = inputs.size()
    xb         = inputs.permute(0,2,3,1).contiguous().view(size[0], size[2], size[3]*size[1])
    token_np   = xb[0].numpy()                    # (S, T)
    inp_fb     = 16 - inp_ib
    q_inp      = 2**(-inp_fb)
    mv_inp     = 2**(inp_ib - 1) - q_inp
    token_np_f = np.clip(np.round(token_np / q_inp) * q_inp, -mv_inp, mv_inp)

    #  Embedding 
    p_emb    = lsq_params.get("embedding",
                               dict(s=0.007, inv_s=143.0, bit_r=5, thd_neg=-31, thd_pos=31))
    W_emb, s_w_emb = load_w_int('embedding.weight', 'embedding.quan_w_fn.s', 'embedding')
    b_emb          = get_bias('embedding.bias')
    s_act_emb      = abs(float(p_emb['s']))

    T       = token_np_f.shape[1]
    x_int_e = np.zeros((S, T), dtype=np.int32)
    for s in range(S):
        xi, _ = lsq_int_np(token_np_f[s], p_emb)
        x_int_e[s] = xi

    x = np.zeros((S, D), dtype=np.float64)
    for s in range(S):
        x[s] = mac_ds_np(W_emb, s_w_emb, x_int_e[s], s_act_emb, b_emb)
    ref_emb = x.copy()

    #  Transformer layers 
    ref_layers = []
    for i in cfg['layer_indices']:
        ai_key = f"layers_{i}_attention_in_proj"
        ao_key = f"layers_{i}_attention_out_proj"

        # Attention in_proj — poids entiers
        p_attn     = lsq_params.get(f"layers.{i}.attention",
                                     dict(s=1.0, inv_s=1.0, bit_r=3, thd_neg=-3, thd_pos=3))
        W_ai, s_w_ai = load_w_int(f'layers.{i}.attention.in_proj_weight',
                                   f'layers.{i}.attention.quan_w_fn.s', ai_key)
        b_ai           = get_bias(f'layers.{i}.attention.in_proj_bias')
        s_act_attn     = abs(float(p_attn['s']))

        x_int_attn = np.zeros((S, D), dtype=np.int32)
        for s in range(S):
            xi, _ = lsq_int_np(x[s], p_attn)
            x_int_attn[s] = xi

        qkv = np.zeros((S, D*3), dtype=np.float64)
        for s in range(S):
            qkv[s] = mac_ds_np(W_ai, s_w_ai, x_int_attn[s], s_act_attn, b_ai)

        # Multi-head attention — float
        ctx = np.zeros((S, D), dtype=np.float64)
        HD  = cfg['head_dim']
        for h in range(cfg['num_heads']):
            Q = qkv[:, h*HD:(h+1)*HD]
            K = qkv[:, D+h*HD:D+(h+1)*HD]
            V = qkv[:, 2*D+h*HD:2*D+(h+1)*HD]
            sc  = Q @ K.T / math.sqrt(HD)
            sc -= sc.max(axis=1, keepdims=True)
            aw  = np.exp(sc); aw /= aw.sum(axis=1, keepdims=True) + 1e-9
            ctx[:, h*HD:(h+1)*HD] = aw @ V

        # Out proj — float MAC (ctx est float)
        W_ao, s_w_ao = load_w_int(f'layers.{i}.attention.out_proj.weight',
                                   f'layers.{i}.attention.quan_w_out_fn.s', ao_key)
        b_ao          = get_bias(f'layers.{i}.attention.out_proj.bias')

        attn_out = np.zeros((S, D), dtype=np.float64)
        for s in range(S):
            attn_out[s] = out_proj_float_np(W_ao, s_w_ao, ctx[s], b_ao)

        # RMSNorm1
        gamma1 = read_gamma(f'layers.{i}.norm1.weight')
        norm1  = rmsnorm_np(x + attn_out, gamma1)

        # FFN
        ff0k, ff1k = fk[0], fk[1]
        ff0_key = f"layers_{i}_feed_forward_{ff0k}"
        ff1_key = f"layers_{i}_feed_forward_{ff1k}"
        p_f0    = lsq_params.get(f"layers.{i}.feed_forward.{ff0k}",
                                  dict(s=0.25, bit_r=3, thd_neg=-3, thd_pos=3))
        p_f1    = lsq_params.get(f"layers.{i}.feed_forward.{ff1k}",
                                  dict(s=0.5,  bit_r=3, thd_neg=-3, thd_pos=3))

        W_ff0, s_w_ff0 = load_w_int(f'layers.{i}.feed_forward.{ff0k}.weight',
                                     f'layers.{i}.feed_forward.{ff0k}.quan_w_fn.s', ff0_key)
        b_ff0           = get_bias(f'layers.{i}.feed_forward.{ff0k}.bias')
        W_ff1, s_w_ff1  = load_w_int(f'layers.{i}.feed_forward.{ff1k}.weight',
                                     f'layers.{i}.feed_forward.{ff1k}.quan_w_fn.s', ff1_key)
        b_ff1           = get_bias(f'layers.{i}.feed_forward.{ff1k}.bias')

        s_act_f0 = abs(float(p_f0['s']))
        s_act_f1 = abs(float(p_f1['s']))

        mid = np.zeros((S, cfg['hid_dim']), dtype=np.float64)
        for s in range(S):
            xi, _ = lsq_int_np(norm1[s], p_f0)
            r      = mac_ds_np(W_ff0, s_w_ff0, xi, s_act_f0, b_ff0)
            mid[s] = np.maximum(r, 0.0)   # ReLU

        ffn_out = np.zeros((S, D), dtype=np.float64)
        for s in range(S):
            xi, _    = lsq_int_np(mid[s], p_f1)
            ffn_out[s] = mac_ds_np(W_ff1, s_w_ff1, xi, s_act_f1, b_ff1)

        # RMSNorm2
        gamma2 = read_gamma(f'layers.{i}.norm2.weight')
        norm2  = rmsnorm_np(norm1 + ffn_out, gamma2)

        ref_layers.append(dict(attn=attn_out, norm1=norm1, ffn=ffn_out, norm2=norm2))
        x = norm2.copy()

    #  Output 
    p_out          = lsq_params.get("output",
                                     dict(s=0.112, bit_r=5, thd_neg=-15, thd_pos=15))
    W_out, s_w_out = load_w_int('output.weight', 'output.quan_w_fn.s', 'output')
    b_out          = get_bias('output.bias')
    s_act_out      = abs(float(p_out['s']))

    raw = np.zeros((S, OUT), dtype=np.float64)
    for s in range(S):
        xi, _   = lsq_int_np(x[s], p_out)
        raw[s]  = mac_ds_np(W_out, s_w_out, xi, s_act_out, b_out)

    ref_mag = np.sqrt(raw[:, :OUT//2]**2 + raw[:, OUT//2:]**2)
    print("=== NUMPY REF EMBEDDING [0..3] ===", ref_emb[0, :4])
    print("=== LSQ emb s=", lsq_params.get('embedding', {}).get('s'), 
        "thd_pos=", lsq_params.get('embedding', {}).get('thd_pos'))
    return token_np, ref_emb, ref_layers, raw, ref_mag
def gen_test_vectors(cfg, lsq_params, sd, outdir, weight_meta, inputs, inp_ib, inp_fb, seed):
    """
    Génère hls_test_vectors.h via forward_hls_np (identique à eval_delayed_scale.py).
    Garantit que ref C++ == ref numpy sans aucune divergence.
    """
    import json as _json
    sys.path.insert(0, cfg['project_root'])
    from Pipeline.eval_delayed_scale import (
        lsq_int_np, mac_ds_np, out_proj_float_np,
        rmsnorm_np, quantize_gamma, quantize_input_apfixed,
        load_hls_weights
    )

    D   = cfg['emb_dim']
    OUT = cfg['output_dim']
    S   = cfg['seq_len']
    NH  = cfg['num_heads']
    HD  = cfg['head_dim']
    layer_indices = cfg['layer_indices']
    ffn_keys      = cfg['ffn_keys']
    ff0, ff1      = ffn_keys[0], ffn_keys[1]

    # Charge poids via load_hls_weights — même logique que eval_delayed_scale
    hw_file = cfg.get('hw_analysis', '')
    if hw_file and Path(hw_file).exists():
        with open(hw_file) as f:
            hw = _json.load(f)
    else:
        from Pipeline.eval_delayed_scale import _build_hw_from_sd
        hw = _build_hw_from_sd(sd)

    W, S_w, B, P, _, _ = load_hls_weights(sd, hw)

    # Input: premier sample, quantifié ap_fixed<16,inp_ib>
    size     = inputs.size()
    xb       = inputs.permute(0,2,3,1).contiguous().view(size[0], size[2], size[3]*size[1])
    token_np = xb[0].numpy()          # (S, T) raw
    token_q  = quantize_input_apfixed(token_np, inp_ib)  # (S, T) quantifié

    T = token_q.shape[1]

    # Embedding
    x_int = np.zeros((S, T), dtype=np.int32)
    for s in range(S):
        xi, _ = lsq_int_np(token_q[s], P['emb'])
        x_int[s] = xi
    s_act_emb = float(P['emb']['s'])
    x = np.zeros((S, D), dtype=np.float64)
    for s in range(S):
        x[s] = mac_ds_np(W['emb'], S_w['emb'], x_int[s], s_act_emb, B['emb'])
    ref_emb = x.copy()

    # Transformer layers
    ref_layers = []
    for i in layer_indices:
        p_attn     = P[f'attn_{i}']
        s_act_attn = float(p_attn['s'])
        x_int_attn = np.zeros((S, D), dtype=np.int32)
        for s in range(S):
            xi, _ = lsq_int_np(x[s], p_attn)
            x_int_attn[s] = xi
        qkv = np.zeros((S, D*3), dtype=np.float64)
        for s in range(S):
            qkv[s] = mac_ds_np(W[f'ai_{i}'], S_w[f'ai_{i}'],
                               x_int_attn[s], s_act_attn, B[f'ai_{i}'])
        ctx = np.zeros((S, D), dtype=np.float64)
        for h in range(NH):
            Q = qkv[:, h*HD:(h+1)*HD]
            K = qkv[:, D+h*HD:D+(h+1)*HD]
            V = qkv[:, 2*D+h*HD:2*D+(h+1)*HD]
            sc  = Q @ K.T / math.sqrt(HD)
            sc -= sc.max(axis=1, keepdims=True)
            aw  = np.exp(sc); aw /= aw.sum(axis=1, keepdims=True) + 1e-9
            ctx[:, h*HD:(h+1)*HD] = aw @ V
        attn_out = np.zeros((S, D), dtype=np.float64)
        for s in range(S):
            attn_out[s] = out_proj_float_np(
                W[f'ao_{i}'], S_w[f'ao_{i}'], ctx[s], B[f'ao_{i}'])

        gamma1 = quantize_gamma(sd[f'layers.{i}.norm1.weight'].float().numpy())
        norm1  = rmsnorm_np(x + attn_out, gamma1)

        p_ff0    = P[f'ff{ff0}_{i}']
        s_act_f0 = float(p_ff0['s'])
        mid = np.zeros((S, W[f'ff{ff0}_{i}'].shape[0]), dtype=np.float64)
        for s in range(S):
            xi, _ = lsq_int_np(norm1[s], p_ff0)
            r = mac_ds_np(W[f'ff{ff0}_{i}'], S_w[f'ff{ff0}_{i}'],
                          xi, s_act_f0, B[f'ff{ff0}_{i}'])
            mid[s] = np.maximum(r, 0.0)

        p_ff1    = P[f'ff{ff1}_{i}']
        s_act_f1 = float(p_ff1['s'])
        ffn_out  = np.zeros((S, D), dtype=np.float64)
        for s in range(S):
            xi, _ = lsq_int_np(mid[s], p_ff1)
            ffn_out[s] = mac_ds_np(W[f'ff{ff1}_{i}'], S_w[f'ff{ff1}_{i}'],
                                   xi, s_act_f1, B[f'ff{ff1}_{i}'])

        gamma2 = quantize_gamma(sd[f'layers.{i}.norm2.weight'].float().numpy())
        x      = rmsnorm_np(norm1 + ffn_out, gamma2)

        ref_layers.append(dict(
            attn=attn_out, norm1=norm1, ffn=ffn_out, norm2=x.copy()))

    # Output
    p_out     = P['out']
    s_act_out = float(p_out['s'])
    raw = np.zeros((S, OUT), dtype=np.float64)
    for s in range(S):
        xi, _ = lsq_int_np(x[s], p_out)
        raw[s] = mac_ds_np(W['out'], S_w['out'], xi, s_act_out, B['out'])

    ref_mag = np.sqrt(raw[:, :OUT//2]**2 + raw[:, OUT//2:]**2)
    print("=== NUMPY REF EMBEDDING [0..3] ===", ref_emb[0, :4])

    # Écrit le .h — token_q déjà quantifié ap_fixed (pas de double quantification)
    lines = [
        f"// Delayed Scaling HLS test vectors | seed={seed}",
        f"// S={S} D={D} OUT={OUT} NL={cfg['num_layers']}",
        f"// REF = forward_hls_np exact (identique a eval_delayed_scale.py)",
        "#pragma once\n",
        f"#define TEST_SEQ_LEN {S}",
        f"#define TEST_EMB_DIM {D}",
        f"#define TEST_OUT_DIM {OUT}\n",
        arr_to_c_float("test_input_flat", token_q.flatten()),
        arr_to_c_float("ref_emb_flat",    ref_emb.flatten()),
    ]
    for i, refs in enumerate(ref_layers):
        lines += [
            arr_to_c_float(f"ref_attn_{i}_flat",  refs['attn'].flatten()),
            arr_to_c_float(f"ref_norm1_{i}_flat", refs['norm1'].flatten()),
            arr_to_c_float(f"ref_ffn_{i}_flat",   refs['ffn'].flatten()),
            arr_to_c_float(f"ref_norm2_{i}_flat", refs['norm2'].flatten()),
        ]
    lines += [
        arr_to_c_float("ref_output_flat", raw.flatten()),
        arr_to_c_float("ref_mag_flat",    ref_mag.flatten()),
    ]
    (outdir / "hls_test_vectors.h").write_text("\n".join(lines))
    print(f"  ok hls_test_vectors.h (DS, seed={seed})")


def gen_testbench(cfg, norm_safes, lsq_params, outdir, inp_tb, inp_ib, weight_meta):
    D   = cfg['emb_dim']
    T   = cfg['token_dim']
    OUT = cfg['output_dim']
    S   = cfg['seq_len']
    NL  = cfg['num_layers']
    fk  = cfg['ffn_keys']
    m_out = weight_meta['output']
    last    = cfg['layer_indices'][-1]
    b_a_out = m_out['b_a']
    s_out     = abs(float(lsq_params.get('output', dict(s=0.112))['s']))
    inv_s_out = 1.0 / s_out if s_out > 0 else 1.0
    thd_out   = lsq_params.get('output', dict(thd_pos=15))['thd_pos']

    check_fn = """
static void check(const char* name, float* hls, const float* ref, int N, float tol) {
    float avg=0, mx=0; int f=0; float floor_v=0.05f;
    for(int j=0;j<N;j++){
        float e=fabsf(hls[j]-ref[j]);
        float r=(fabsf(ref[j])>floor_v?e/fabsf(ref[j]):e/floor_v);
        avg+=r; if(r>mx)mx=r; if(r>tol)f++;
    }
    avg/=N;
    printf("%-28s avg_rel=%.4f max_rel=%.4f fail=%d/%d %s\\n",
           name,avg,mx,f,N,f==0?"ok":"WARN");
    printf("  HLS[0..3]: %+.5f %+.5f %+.5f %+.5f\\n",hls[0],hls[1],hls[2],hls[3]);
    printf("  REF[0..3]: %+.5f %+.5f %+.5f %+.5f\\n",ref[0],ref[1],ref[2],ref[3]);
}
static void check_abs(const char* name, float* hls, const float* ref, int N, float tol) {
    float avg=0, mx=0; int f=0;
    for(int j=0;j<N;j++){
        float e=fabsf(hls[j]-ref[j]);
        avg+=e; if(e>mx)mx=e; if(e>tol)f++;
    }
    avg/=N;
    printf("%-28s avg_abs=%.4f max_abs=%.4f fail=%d/%d %s\\n",
           name,avg,mx,f,N,f==0?"ok":"WARN");
    printf("  HLS[0..3]: %+.5f %+.5f %+.5f %+.5f\\n",hls[0],hls[1],hls[2],hls[3]);
    printf("  REF[0..3]: %+.5f %+.5f %+.5f %+.5f\\n",ref[0],ref[1],ref[2],ref[3]);
}
static float cosine_sim(const float* a, const float* b, int N) {
    float dot=0, na=0, nb=0;
    for(int j=0;j<N;j++){dot+=a[j]*b[j]; na+=a[j]*a[j]; nb+=b[j]*b[j];}
    return dot/(sqrtf(na)*sqrtf(nb)+1e-9f);
}
"""

    # TEST 1: blocs vraiment isolés
    # Chaque attention reçoit sa ref numpy directement
    # Chaque norm reçoit le résidu calculé avec les refs numpy (pas les sorties HLS)
    # Chaque FFN reçoit ref_norm1 directement
    isolated_blocks = []
    for i in cfg['layer_indices']:
        ref_in = "ref_emb_flat" if i == 0 else f"ref_norm2_{i-1}_flat"
        isolated_blocks.append(f"""
    //  Layer {i} vraiment isolé 
    {{
        // Attn: reçoit ref_in directement
        float ref_in_{i}[SEQ_LEN][EMB_DIM];
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            ref_in_{i}[si][j] = {ref_in}[si*EMB_DIM+j];
        float attn_out_{i}[SEQ_LEN][EMB_DIM];
        attention_layer_{i}(ref_in_{i}, attn_out_{i});
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = attn_out_{i}[si][j];
        check("L{i}_attn", buf, ref_attn_{i}_flat, SEQ_LEN*EMB_DIM, 0.15f);

        // Norm1: résidu avec refs numpy (attn HLS + ref_in numpy)
        float norm1_in_{i}[SEQ_LEN][EMB_DIM], norm1_out_{i}[SEQ_LEN][EMB_DIM];
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            norm1_in_{i}[si][j] = {ref_in}[si*EMB_DIM+j] + ref_attn_{i}_flat[si*EMB_DIM+j];
        rmsnorm_layer_{i}_1(norm1_in_{i}, norm1_out_{i});
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = norm1_out_{i}[si][j];
        check("L{i}_norm1", buf, ref_norm1_{i}_flat, SEQ_LEN*EMB_DIM, 0.15f);

        // FFN: reçoit ref_norm1 numpy directement (pas HLS norm1)
        float ffn_in_{i}[SEQ_LEN][EMB_DIM];
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            ffn_in_{i}[si][j] = ref_norm1_{i}_flat[si*EMB_DIM+j];
        float ffn_out_{i}[SEQ_LEN][EMB_DIM];
        ffn_block_{i}(ffn_in_{i}, ffn_out_{i});
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = ffn_out_{i}[si][j];
        check_abs("L{i}_ffn", buf, ref_ffn_{i}_flat, SEQ_LEN*EMB_DIM, 2.0f);

        // Norm2: résidu avec refs numpy (ffn HLS + ref_norm1 numpy)
        float norm2_in_{i}[SEQ_LEN][EMB_DIM], norm2_out_{i}[SEQ_LEN][EMB_DIM];
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            norm2_in_{i}[si][j] = ref_norm1_{i}_flat[si*EMB_DIM+j]
                                 + ref_ffn_{i}_flat[si*EMB_DIM+j];
        rmsnorm_layer_{i}_2(norm2_in_{i}, norm2_out_{i});
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = norm2_out_{i}[si][j];
        check("L{i}_norm2", buf, ref_norm2_{i}_flat, SEQ_LEN*EMB_DIM, 0.15f);
    }}""")

    # run_layer helpers pour TEST 2
    run_layers = []
    for i in cfg['layer_indices']:
        run_layers.append(f"""
static void run_layer_{i}(float x[SEQ_LEN][EMB_DIM]) {{
    float attn_out[SEQ_LEN][EMB_DIM];
    attention_layer_{i}(x, attn_out);
    float norm1_in[SEQ_LEN][EMB_DIM], norm1_out[SEQ_LEN][EMB_DIM];
    for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
        norm1_in[si][j] = x[si][j] + attn_out[si][j];
    rmsnorm_layer_{i}_1(norm1_in, norm1_out);
    float ffn_out[SEQ_LEN][EMB_DIM];
    ffn_block_{i}(norm1_out, ffn_out);
    float norm2_in[SEQ_LEN][EMB_DIM];
    for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
        norm2_in[si][j] = norm1_out[si][j] + ffn_out[si][j];
    rmsnorm_layer_{i}_2(norm2_in, x);
}}""")

    chain_calls = ""
    for i in cfg['layer_indices']:
        chain_calls += f"""
        run_layer_{i}(xc);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = xc[si][j];
        check("chain_L{i}", buf, ref_norm2_{i}_flat, SEQ_LEN*EMB_DIM, 0.50f);"""

    fn_decls = "\n".join([
        f"void attention_layer_{i}(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]);"
        f" void rmsnorm_layer_{i}_1(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]);"
        f" void ffn_block_{i}(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]);"
        f" void rmsnorm_layer_{i}_2(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]);"
        for i in cfg['layer_indices']
    ])

    (outdir / "transformer_top_tb.cpp").write_text(
f"""// Delayed Scaling Testbench v4 | S={S} T={T} D={D} NL={NL}
// TEST 1: Blocs vraiment isolés — chaque bloc reçoit la ref numpy comme input
//         Norm reçoit résidu avec refs numpy, FFN reçoit ref_norm1 numpy
// TEST 2: Chaîne HLS complète — mesure accumulation réelle
// TEST 3: E2E transformer_top — validation finale avec cosine similarity
#include <stdio.h>
#include <math.h>
#include "transformer_top.h"
#include "hls_test_vectors.h"
#include "ap_int.h"

void embedding_layer(input_t input[SEQ_LEN][TOKEN_DIM], float output[SEQ_LEN][EMB_DIM]);
void transformer_top(input_t input_flat[{S*T}], float output_flat[{S*OUT}]);
{fn_decls}
void output_layer(float input[SEQ_LEN][EMB_DIM], ap_int<{b_a_out}> x_int[SEQ_LEN][EMB_DIM],
                  float s_act_out, float output_arr[SEQ_LEN][OUTPUT_DIM]);

{check_fn}
{"".join(run_layers)}

static void apply_output(float x[SEQ_LEN][EMB_DIM], float out[SEQ_LEN][OUTPUT_DIM]) {{
    ap_int<{b_a_out}> xi[SEQ_LEN][EMB_DIM];
    for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++) {{
        float _v = roundf(x[si][j] * {inv_s_out:.8f}f);
        _v = _v < {float(-thd_out):.1f}f ? {float(-thd_out):.1f}f
           : (_v > {float(thd_out):.1f}f ? {float(thd_out):.1f}f : _v);
        xi[si][j] = (ap_int<{b_a_out}>)_v;
    }}
    output_layer(x, xi, {s_out:.10f}f, out);
}}

int main() {{
    input_t input_flat[{S*T}];
    float   output_flat[{S*OUT}];
    float   x[SEQ_LEN][EMB_DIM];
    float   buf[SEQ_LEN*EMB_DIM > SEQ_LEN*OUTPUT_DIM ?
                SEQ_LEN*EMB_DIM : SEQ_LEN*OUTPUT_DIM];

    for(int i=0;i<{S*T};i++) input_flat[i]=(input_t)test_input_flat[i];

    printf("\\n=== Delayed Scaling CSIM v4 | S=%d T=%d D=%d NL=%d ===\\n",
           SEQ_LEN, TOKEN_DIM, EMB_DIM, NUM_LAYERS);

    //  TEST 1: Blocs vraiment isolés 
    printf("\\n=== TEST 1: Blocs isolés (ref numpy comme input) ===\\n");
    {{
        // Embedding
        input_t in2d[SEQ_LEN][TOKEN_DIM];
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<TOKEN_DIM;j++)
            in2d[si][j] = input_flat[si*TOKEN_DIM+j];
        embedding_layer(in2d, x);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = x[si][j];
        check("embedding", buf, ref_emb_flat, SEQ_LEN*EMB_DIM, 0.15f);
    }}
{"".join(isolated_blocks)}
    // Output isolé
    {{
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            x[si][j] = ref_norm2_{last}_flat[si*EMB_DIM+j];
        float out2d[SEQ_LEN][OUTPUT_DIM];
        apply_output(x, out2d);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<OUTPUT_DIM;j++)
            buf[si*OUTPUT_DIM+j] = out2d[si][j];
        check_abs("output_isolé", buf, ref_output_flat, SEQ_LEN*OUTPUT_DIM, 0.5f);
    }}

    //  TEST 2: Chaîne HLS 
    printf("\\n=== TEST 2: Chaîne HLS (accumulation) ===\\n");
    {{
        float xc[SEQ_LEN][EMB_DIM];
        input_t in2d[SEQ_LEN][TOKEN_DIM];
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<TOKEN_DIM;j++)
            in2d[si][j] = input_flat[si*TOKEN_DIM+j];
        embedding_layer(in2d, xc);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = xc[si][j];
        check("chain_emb", buf, ref_emb_flat, SEQ_LEN*EMB_DIM, 0.15f);
{chain_calls}
        float out2d[SEQ_LEN][OUTPUT_DIM];
        apply_output(xc, out2d);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<OUTPUT_DIM;j++)
            buf[si*OUTPUT_DIM+j] = out2d[si][j];
        check_abs("chain_output", buf, ref_output_flat, SEQ_LEN*OUTPUT_DIM, 2.0f);
    }}

    //  TEST 3: E2E transformer_top 
    printf("\\n=== TEST 3: E2E transformer_top ===\\n");
    transformer_top(input_flat, output_flat);

    // Cosine similarity par user — ce qui compte pour le precoding
    printf("  Cosine similarity HLS vs ref (direction du beamforming):\\n");
    int all_ok = 1;
    for(int si=0;si<SEQ_LEN;si++) {{
        float cos = cosine_sim(
            output_flat + si*OUTPUT_DIM,
            ref_output_flat + si*OUTPUT_DIM,
            OUTPUT_DIM);
        int ok = (cos > 0.80f);
        if(!ok) all_ok=0;
        printf("    user[%d]: cos_sim=%+.4f %s\\n", si, cos, ok?"OK":"WARN");
    }}
    printf("  E2E: %s\\n", all_ok?"PASS":"FAIL");

    // Magnitude E2E
    printf("  Magnitude HLS vs ref:\\n");
    for(int si=0;si<SEQ_LEN;si++) {{
        float mag_hls=0, mag_ref=0;
        for(int k=0;k<OUTPUT_DIM/2;k++) {{
            float rh=output_flat[si*OUTPUT_DIM+k];
            float ih=output_flat[si*OUTPUT_DIM+k+OUTPUT_DIM/2];
            float rr=ref_output_flat[si*OUTPUT_DIM+k];
            float ir=ref_output_flat[si*OUTPUT_DIM+k+OUTPUT_DIM/2];
            mag_hls += rh*rh+ih*ih;
            mag_ref += rr*rr+ir*ir;
        }}
        printf("    user[%d]: |HLS|=%.3f |REF|=%.3f ratio=%.3f\\n",
               si, sqrtf(mag_hls), sqrtf(mag_ref),
               sqrtf(mag_hls)/(sqrtf(mag_ref)+1e-9f));
    }}
    return 0;
}}
""")
    print(f"  ok transformer_top_tb.cpp v4 (S={S} T={T} D={D} NL={NL})")# ==============================================================================
# Input format calibration
# ==============================================================================

def compute_input_format(token_np):
    max_val  = float(np.abs(token_np).max()) * 1.1
    int_bits = max(int(np.ceil(np.log2(max_val + 1e-8))) + 1, 2)
    frac_bits = 16 - int_bits
    print(f"  Input format: ap_fixed<16,{int_bits}> "
          f"(max={max_val:.4f} res={2**(-frac_bits):.7f})")
    return 16, int_bits, frac_bits


# ==============================================================================
# Main
# ==============================================================================

def main():
    cli    = parse_args()
    outdir = Path(cli.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  HLS Delayed Scaling Generator")
    print(f"{'='*60}")
    print(f"  Output : {outdir}")
    print(f"  Seed   : {cli.seed}")

    # Load checkpoint
    print(f"\n[1/7] Loading checkpoint...")
    ckpt = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    sd   = ckpt.get("model_state_dict", ckpt)
    sd   = {k.replace("module.", ""): v for k, v in sd.items()}
    sd   = {k: v.reshape(1) if v.shape == torch.Size([]) else v
            for k, v in sd.items()}

    # hw_analysis
    print(f"\n[2/7] Loading hw_analysis...")
    with open(cli.hw_analysis) as f:
        hw = json.load(f)
    bit_widths = hw["bit_widths"]

    # Architecture
    print(f"\n[3/7] Detecting architecture...")
    cfg = detect_architecture(sd, cli.project_root, cli.scenario, cli.seed)
    cfg['hw_analysis'] = cli.hw_analysis

    # LSQ activation params
    print(f"\n[4/7] Collecting LSQ activation params...")
    lsq_params, lsq_model = collect_lsq_act_params(sd, cfg)

    # Weight files — Delayed Scaling
    print(f"\n[5/7] Generating weight files (Delayed Scaling)...")
    fk = cfg['ffn_keys']
    weight_meta = {}

    def process_layer(layer_name, w, b, s, w_bit, a_bit):
        safe = layer_name.replace(".", "_")
        meta = gen_weight_files_ds(layer_name, w, b, s, w_bit, a_bit, outdir)
        weight_meta[safe] = meta

    for layer_name, bits in bit_widths.items():
        w_bit, a_bit = bits["weight"], bits["activation"]

        if layer_name + ".in_proj_weight" in sd:
            # MHA: in_proj and out_proj separate
            for suffix, proj, w_key, b_key, s_fn in [
                ("_in_proj",  "in",
                 layer_name + ".in_proj_weight",
                 layer_name + ".in_proj_bias",
                 lambda ln: get_lsq_weight_scale(sd, ln)),
                ("_out_proj", "out",
                 layer_name + ".out_proj.weight",
                 layer_name + ".out_proj.bias",
                 lambda ln: get_lsq_weight_scale(sd, ln, proj_type='out')),
            ]:
                w = sd[w_key].float().numpy()
                b = sd[b_key].float().numpy() if b_key in sd else None
                s = get_lsq_weight_scale(sd, layer_name,
                                         proj_type=None if suffix=="_in_proj" else "out")
                if s is None: s = np.ones((w.shape[0], 1)) * 0.1
                process_layer(layer_name + suffix, w, b, s, w_bit, a_bit)
            continue

        wk = layer_name + ".weight"
        if wk not in sd:
            print(f"  [SKIP] {layer_name}")
            continue
        w = sd[wk].float().numpy()
        b = sd.get(layer_name + ".bias", None)
        if b is not None: b = b.float().numpy()
        s = get_lsq_weight_scale(sd, layer_name)
        if s is None: s = np.ones((w.shape[0], 1)) * 0.1
        process_layer(layer_name, w, b, s, w_bit, a_bit)

    # RMSNorm files
    norm_safes = {}
    for i in cfg['layer_indices']:
        for ni, key in [(1, f"layers.{i}.norm1.weight"),
                        (2, f"layers.{i}.norm2.weight")]:
            if key in sd:
                safe = gen_rmsnorm_files(
                    key.replace(".weight", ""), sd[key].float().numpy(), outdir)
                norm_safes[(i, ni)] = safe
                print(f"  ok {key.replace('.weight','')}")

    # Output layer weight
    if "output.weight" in sd:
        w  = sd["output.weight"].float().numpy()
        b  = sd.get("output.bias", None)
        if b is not None: b = b.float().numpy()
        ob = bit_widths.get("output", {"weight": 8, "activation": 8})
        s  = get_lsq_weight_scale(sd, "output")
        if s is None: s = np.ones((w.shape[0], 1)) * 0.1
        process_layer("output", w, b, s, ob["weight"], ob["activation"])

    # Sample + input format
    print(f"\n[5b] Loading sample and calibrating input format...")
    inputs, token_np = load_sample(cli.project_root, cli.scenario, cli.seed)
    inp_tb, inp_ib, inp_fb = compute_input_format(token_np)

    # HLS blocks
    print(f"\n[6/7] Generating HLS blocks...")
    gen_embedding_block(cfg, outdir, inp_tb, inp_ib, lsq_params, weight_meta)
    for i in cfg['layer_indices']:
        gen_attention_block(i, cfg, lsq_params, outdir, weight_meta)
        gen_ffn_block(i, cfg, lsq_params, outdir, weight_meta)
        gen_rmsnorm_block(i, 1, norm_safes[(i, 1)], cfg, outdir)
        gen_rmsnorm_block(i, 2, norm_safes[(i, 2)], cfg, outdir)
    gen_output_block(cfg, outdir, weight_meta)
    gen_transformer_top(cfg, norm_safes, lsq_params, outdir, inp_tb, inp_ib, weight_meta)

    # Test vectors + testbench
    print(f"\n[7/7] Generating test vectors and testbench...")
    gen_test_vectors(cfg, lsq_params, sd, outdir, weight_meta,
                     inputs, inp_ib, inp_fb, cli.seed)
    gen_testbench(cfg, norm_safes, lsq_params, outdir, inp_tb, inp_ib, weight_meta)

    cpp_files = sorted(outdir.glob("*.cpp"))
    h_files   = sorted(outdir.glob("*.h"))
    print(f"\n{'='*60}")
    print(f"  Done: {outdir}")
    print(f"  {len(cpp_files)} .cpp  |  {len(h_files)} .h")
    print(f"  Top: transformer_top")
    print(f"  Input:  input_t input_flat[{cfg['seq_len']*cfg['token_dim']}]")

    print(f"  Output: float output_flat[{cfg['seq_len']*cfg['output_dim']}]")
    print(f"  Latency pragmas: PIPELINE II=1 on all inner loops")
    print(f"  LUT/DSP: fabric when b_w+b_a<=10, dsp otherwise")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()