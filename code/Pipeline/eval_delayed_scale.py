#!/usr/bin/env python3
"""
eval_hls_numpy.py
Simule EXACTEMENT le forward HLS Delayed Scaling tel que généré par
generate_hls_project.py — même arithmétique, mêmes thresholds entiers,
même out_proj float — et compare SR vs PyTorch quantifié.

Si SR_numpy ≈ SR_pytorch (<5%) → le HLS C++ est correct.
Si SR_numpy << SR_pytorch → bug dans generate_hls ou dégradation hardware réelle.

Usage:
  python eval_hls_numpy.py --scenario stecath --n_batches 30 \
    --configs lambda_5:/export/tmp/sala/results_fpga/FPGA_medium_fpga_stecath/phase2_quant/lambda_5.0/checkpoints/best_model.pth
"""

import sys, math, json, argparse
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = '/users/sala/Documents/ELE6310/ELE6310E'
sys.path.insert(0, PROJECT_ROOT)

from Pipeline.input_args import input_args
from Pipeline.utils import utils_
from Pipeline.Learner import Learner
import quan

# ==============================================================================
# Primitives — copie exacte de generate_hls_project.py
# ==============================================================================

def lsq_int_np(arr, p):
    """
    Quantifie arr en entier avec thd ENTIER (comme ap_int<b> en HLS).
    p = dict(s=float, thd_pos=int, thd_neg=int)
    thd_pos = 2^ceil(bit) - 1
    """
    s     = abs(float(p['s']))
    inv_s = 1.0 / s if s > 0 else 1.0
    thd   = int(p['thd_pos'])  # entier pur — c'est ce que ap_int<b> peut stocker
    v = np.clip(
        np.round(np.array(arr, dtype=np.float64) * inv_s),
        -thd, thd
    )
    return v.astype(np.int32), s


def mac_ds_np(w_int, s_w, x_int, s_act, bias):
    """
    MAC Delayed Scaling exact:
      acc = W_int @ x_int  (entier → int64)
      y   = acc * s_w[i] * s_act + B[i]  (float par neurone)
    C'est exactement ce que mac_loop_ds génère en HLS.
    """
    acc    = w_int.astype(np.int64) @ x_int.astype(np.int64)  # (out,) int64
    result = acc.astype(np.float64) * s_w.astype(np.float64) * float(s_act)
    if bias is not None:
        result += bias.astype(np.float64)
    return result


def out_proj_float_np(w_int, s_w, ctx, bias):
    """
    Out proj attention: ctx est float (post-softmax).
    HLS génère: acc += W[r][c] * s_w[r] * ctx[c]  (float MAC)
    = (W_int * s_w) @ ctx — pas d'entier ici.
    """
    w_dq = w_int.astype(np.float64) * s_w.reshape(-1, 1)
    result = w_dq @ ctx.astype(np.float64)
    if bias is not None:
        result += bias.astype(np.float64)
    return result


def rmsnorm_np(x, gamma_q):
    """
    RMSNorm avec gamma quantifié ap_fixed<8,4> comme dans gen_rmsnorm_files.
    gamma_q = clip(round(gamma / 2^-4) * 2^-4, -(2^3 - 2^-4), 2^3 - 2^-4)
    """
    out = np.zeros_like(x, dtype=np.float64)
    for s in range(x.shape[0]):
        ms    = float(np.mean(x[s] ** 2)) + 1e-5
        out[s] = x[s] / np.sqrt(ms) * gamma_q.astype(np.float64)
    return out


def quantize_gamma(gamma):
    """Reproduit gen_rmsnorm_files: gamma stocké en ap_fixed<8,4>."""
    q  = 2**(-4)
    mv = 2**3 - q
    return np.clip(np.round(gamma / q) * q, -mv, mv)


def quantize_input_apfixed(token_np, inp_ib):
    """
    Quantifie l'input en ap_fixed<16,inp_ib> comme dans transformer_top.cpp.
    inp_ib détecté par compute_input_format dans generate_hls_project.py.
    """
    inp_fb = 16 - inp_ib
    q      = 2**(-inp_fb)
    mv     = 2**(inp_ib - 1) - q
    return np.clip(np.round(token_np / q) * q, -mv, mv)


# ==============================================================================
# Chargement poids — même logique que generate_hls_project.py
# ==============================================================================
def load_hls_weights(sd, hw):
    bit_widths = hw['bit_widths']

    def get_w(key):
        return sd[key].float().numpy() if key in sd else None

    def load_layer(w_key, s_key, b_key, b_w):
        w = get_w(w_key)
        if w is None: return None, None, None
        s = np.abs(sd[s_key].float().numpy()).flatten() if s_key in sd \
            else np.ones(w.shape[0]) * 0.1
        b = sd[b_key].float().numpy() if b_key in sd else None
        thd   = 2**b_w - 1
        w_int = np.clip(np.round(w / s.reshape(-1, 1)), -thd, thd).astype(np.int32)
        return w_int, s, b

    def act_params(s_key, bit_key):
        # Lit bit directement depuis checkpoint — round plus fidèle à PyTorch
        bit = float(sd[bit_key].float()) if bit_key in sd else 8.0
        b_a = max(2, round(bit))
        thd = 2**b_a - 1
        s_val = abs(float(sd[s_key].float())) if s_key in sd else 1.0
        return dict(s=s_val, thd_pos=thd, thd_neg=-thd)

    W, S_w, B, P = {}, {}, {}, {}

    layer_indices = sorted(set(
        int(k.split('.')[1]) for k in sd
        if k.startswith('layers.') and k.split('.')[1].isdigit()))
    ffn_keys = sorted(set(
        int(k.split('.')[3]) for k in sd
        if k.startswith('layers.0.feed_forward.') and
        k.split('.')[3].isdigit() and 'weight' in k))

    # Embedding
    bw = bit_widths.get('embedding', {}).get('weight', 2)
    W['emb'], S_w['emb'], B['emb'] = load_layer(
        'embedding.weight', 'embedding.quan_w_fn.s', 'embedding.bias', bw)
    P['emb'] = act_params('embedding.quan_a_fn.s', 'embedding.quan_a_fn.bit')

    for i in layer_indices:
        ln_attn = f'layers.{i}.attention'
        bw_ai = bit_widths.get(ln_attn, {}).get('weight', 2)

        W[f'ai_{i}'], S_w[f'ai_{i}'], B[f'ai_{i}'] = load_layer(
            f'{ln_attn}.in_proj_weight',
            f'{ln_attn}.quan_w_fn.s',
            f'{ln_attn}.in_proj_bias', bw_ai)
        P[f'attn_{i}'] = act_params(
            f'{ln_attn}.quan_a_fn.s', f'{ln_attn}.quan_a_fn.bit')

        W[f'ao_{i}'], S_w[f'ao_{i}'], B[f'ao_{i}'] = load_layer(
            f'{ln_attn}.out_proj.weight',
            f'{ln_attn}.quan_w_out_fn.s',
            f'{ln_attn}.out_proj.bias', bw_ai)

        for ni in [1, 2]:
            key = f'layers.{i}.norm{ni}.weight'
            W[f'norm{ni}_{i}'] = quantize_gamma(sd[key].float().numpy()) \
                if key in sd else np.ones(sd['embedding.weight'].shape[0])

        for fk in ffn_keys:
            ln_ff = f'layers.{i}.feed_forward.{fk}'
            bw_ff = bit_widths.get(f'layers.{i}.feed_forward.{fk}', {}).get('weight', 2)
            W[f'ff{fk}_{i}'], S_w[f'ff{fk}_{i}'], B[f'ff{fk}_{i}'] = load_layer(
                f'{ln_ff}.weight', f'{ln_ff}.quan_w_fn.s', f'{ln_ff}.bias', bw_ff)
            P[f'ff{fk}_{i}'] = act_params(
                f'{ln_ff}.quan_a_fn.s', f'{ln_ff}.quan_a_fn.bit')

    # Output
    bw_out = bit_widths.get('output', {}).get('weight', 4)
    W['out'], S_w['out'], B['out'] = load_layer(
        'output.weight', 'output.quan_w_fn.s', 'output.bias', bw_out)
    P['out'] = act_params('output.quan_a_fn.s', 'output.quan_a_fn.bit')

    return W, S_w, B, P, layer_indices, ffn_keys
# ==============================================================================
# Forward HLS numpy — reproduit EXACTEMENT transformer_top.cpp
# ==============================================================================

def forward_hls_np(ch_norm_np, W, S_w, B, P, layer_indices, ffn_keys,
                   num_heads, head_dim, output_dim, inp_ib):
    """
    Forward pass numpy identique au HLS généré.
    ch_norm_np: (2, 4, 64) numpy float
    Retourne: complex tensor (output_dim//2, seq_len)
    """
    D   = W['emb'].shape[0]      # emb_dim
    S   = ch_norm_np.shape[1]    # seq_len = 4
    NH  = num_heads
    HD  = head_dim
    OUT = output_dim
    ff0, ff1 = ffn_keys[0], ffn_keys[1]

    # Construction token — identique à transformer_top.cpp reshape
    token_np = ch_norm_np.transpose(1, 2, 0).reshape(S, -1)  # (4, 128)

    # Quantification input ap_fixed<16,inp_ib>
    token_q = quantize_input_apfixed(token_np, inp_ib)
    
    # ── Embedding ────────────────────────────────────────────────────────────
    token_dim = token_q.shape[1]  # 128 (≠ emb_dim=192)
    x_int = np.zeros((S, token_dim), dtype=np.int32)
    for s in range(S):
        xi, _ = lsq_int_np(token_q[s], P['emb'])
        x_int[s] = xi
    s_act_emb = float(P['emb']['s'])

    x = np.zeros((S, D), dtype=np.float64)
    for s in range(S):
        x[s] = mac_ds_np(W['emb'], S_w['emb'], x_int[s], s_act_emb, B['emb'])

    # ── Transformer layers ───────────────────────────────────────────────────
    for i in layer_indices:
        # — Attention in_proj —
        p_attn    = P[f'attn_{i}']
        s_act_attn = float(p_attn['s'])

        x_int_attn = np.zeros((S, D), dtype=np.int32)
        for s in range(S):
            xi, _ = lsq_int_np(x[s], p_attn)
            x_int_attn[s] = xi

        qkv = np.zeros((S, D * 3), dtype=np.float64)
        for s in range(S):
            qkv[s] = mac_ds_np(
                W[f'ai_{i}'], S_w[f'ai_{i}'],
                x_int_attn[s], s_act_attn, B[f'ai_{i}'])

        # — Multi-head attention (float — softmax) —
        ctx = np.zeros((S, D), dtype=np.float64)
        for h in range(NH):
            Q = qkv[:, h*HD:(h+1)*HD]
            K = qkv[:, D + h*HD:D + (h+1)*HD]
            V = qkv[:, 2*D + h*HD:2*D + (h+1)*HD]
            sc = Q @ K.T / math.sqrt(HD)
            sc -= sc.max(axis=1, keepdims=True)  # stabilité numérique
            aw  = np.exp(sc)
            aw /= aw.sum(axis=1, keepdims=True) + 1e-9
            ctx[:, h*HD:(h+1)*HD] = aw @ V

        # — Out proj: float MAC (ctx est float post-softmax) —
        attn_out = np.zeros((S, D), dtype=np.float64)
        for s in range(S):
            attn_out[s] = out_proj_float_np(
                W[f'ao_{i}'], S_w[f'ao_{i}'], ctx[s], B[f'ao_{i}'])

        # — Residual + RMSNorm1 —
        norm1 = rmsnorm_np(x + attn_out, W[f'norm1_{i}'])

        # — FFN layer 0 + ReLU —
        p_ff0    = P[f'ff{ff0}_{i}']
        s_act_f0 = float(p_ff0['s'])
        mid = np.zeros((S, W[f'ff{ff0}_{i}'].shape[0]), dtype=np.float64)
        for s in range(S):
            xi, _ = lsq_int_np(norm1[s], p_ff0)
            r = mac_ds_np(W[f'ff{ff0}_{i}'], S_w[f'ff{ff0}_{i}'],
                          xi, s_act_f0, B[f'ff{ff0}_{i}'])
            mid[s] = np.maximum(r, 0.0)  # ReLU

        # — FFN layer 1 —
        p_ff1    = P[f'ff{ff1}_{i}']
        s_act_f1 = float(p_ff1['s'])
        ffn_out = np.zeros((S, D), dtype=np.float64)
        for s in range(S):
            xi, _ = lsq_int_np(mid[s], p_ff1)
            ffn_out[s] = mac_ds_np(W[f'ff{ff1}_{i}'], S_w[f'ff{ff1}_{i}'],
                                   xi, s_act_f1, B[f'ff{ff1}_{i}'])

        # — Residual + RMSNorm2 —
        x = rmsnorm_np(norm1 + ffn_out, W[f'norm2_{i}'])

    # ── Output layer ─────────────────────────────────────────────────────────
    p_out    = P['out']
    s_act_out = float(p_out['s'])
    raw = np.zeros((S, OUT), dtype=np.float64)
    for s in range(S):
        xi, _ = lsq_int_np(x[s], p_out)
        raw[s] = mac_ds_np(W['out'], S_w['out'], xi, s_act_out, B['out'])

    # Format sortie: complex (OUT//2, S)
    half = OUT // 2
    return torch.complex(
        torch.tensor(raw[:, :half].T, dtype=torch.float32),
        torch.tensor(raw[:, half:].T, dtype=torch.float32),
    )


# ==============================================================================
# Loader PyTorch quantifié
# ==============================================================================

def load_model_quantized(ckpt_path, sd, layer_indices, ffn_keys):
    from Pipeline.Transformer_FPGA import StackedTransformer

    emb_dim   = sd['embedding.weight'].shape[0]
    token_dim = sd['embedding.weight'].shape[1]
    hid_dim   = sd[f'layers.0.feed_forward.{ffn_keys[0]}.weight'].shape[0]
    out_dim   = sd['output.weight'].shape[0]
    num_heads = None
    for nh in [6, 4, 8, 2, 1]:
        if emb_dim % nh == 0:
            num_heads = nh; break

    class DotDict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__

    qcfg = DotDict(
        act=DotDict(mode='lsq', bit=8, per_channel=False,
                    symmetric=False, all_positive=False),
        weight=DotDict(mode='lsq', bit=8, per_channel=True,
                       symmetric=False, all_positive=False),
        excepts=DotDict()
    )
    tcfg = dict(num_layers=len(layer_indices), embedding_dim=emb_dim,
                num_heads=num_heads, hidden_dim=hid_dim,
                token_dim=token_dim, dropout=0)
    model = StackedTransformer(**tcfg)
    rep, _ = quan.find_modules_to_quantize(model, qcfg)
    model  = quan.replace_module_by_names(model, rep)
    sd2    = {k: v.reshape(1) if v.shape == torch.Size([]) else v
              for k, v in sd.items()}
    model.load_state_dict(sd2, strict=False)
    model.eval()
    return model, emb_dim, num_heads, hid_dim // emb_dim * emb_dim


# ==============================================================================
# Main
# ==============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scenario',  default='stecath')
    p.add_argument('--n_batches', type=int, default=30)
    p.add_argument('--configs',   nargs='+', required=True,
                   help='name:ckpt_path pairs, ex: lambda_5:/path/to/best_model.pth')
    p.add_argument('--hw_analysis_dir', default=None,
                   help='Dossier contenant hw_analysis_lambda_X.json '
                        '(auto-détecté depuis ckpt_path si absent)')
    cli, _ = p.parse_known_args()

    configs = []
    for s in cli.configs:
        name, ckpt = s.split(':', 1)
        configs.append(dict(name=name, ckpt=ckpt))

    print(f"\n{'='*60}")
    print(f"  eval_hls_numpy | scenario={cli.scenario} | n_batches={cli.n_batches}")
    print(f"  Simule EXACTEMENT le HLS C++ généré par generate_hls_project.py")
    print(f"{'='*60}")

    # Loader dataset
    saved = sys.argv[:]
    sys.argv = [sys.argv[0], '--scenario', cli.scenario]
    loader_args = input_args().args
    loader_args.scenario = [cli.scenario]
    sys.argv = saved
    _, loader = utils_(loader_args).Data_Load()
    learner = Learner(loader_args)

    all_results = {}

    for cfg in configs:
        print(f"\n{'─'*60}")
        print(f"  Config: {cfg['name']}")
        print(f"  Ckpt  : {cfg['ckpt']}")

        # Charge checkpoint
        sd_raw = torch.load(cfg['ckpt'], map_location='cpu', weights_only=False)
        sd     = sd_raw.get('model_state_dict', sd_raw)
        sd     = {k.replace('module.', ''): v for k, v in sd.items()}
        sd     = {k: v.reshape(1) if v.shape == torch.Size([]) else v
                  for k, v in sd.items()}

        # Auto-detect hw_analysis
        ckpt_path = Path(cfg['ckpt'])
        if cli.hw_analysis_dir:
            hw_dir = Path(cli.hw_analysis_dir)
        else:
            # ex: .../phase2_quant/lambda_5.0/checkpoints/best_model.pth
            # hw_analysis est dans .../hls_analysis/hw_analysis_lambda_5.0.json
            run_dir = ckpt_path.parent.parent.parent.parent
            hw_dir  = run_dir / 'hls_analysis'

        # Trouve le fichier hw_analysis correspondant
        lam_str = ckpt_path.parent.parent.name  # ex: lambda_5.0
        hw_file = hw_dir / f'hw_analysis_{lam_str}.json'
        if not hw_file.exists():
            # Essai glob
            import glob
            candidates = list(hw_dir.glob(f'hw_analysis_{lam_str}*.json'))
            if candidates:
                hw_file = candidates[0]
            else:
                print(f"  [WARN] hw_analysis non trouvé: {hw_file}")
                print(f"         Utilisation de bit_widths par défaut (ceil des bits LSQ)")
                # Construit bit_widths depuis le checkpoint directement
                hw = _build_hw_from_sd(sd)
        if hw_file.exists():
            with open(hw_file) as f:
                hw = json.load(f)
            print(f"  hw_analysis: {hw_file.name}")

        # Architecture
        layer_indices = sorted(set(
            int(k.split('.')[1]) for k in sd
            if k.startswith('layers.') and k.split('.')[1].isdigit()))
        ffn_keys = sorted(set(
            int(k.split('.')[3]) for k in sd
            if k.startswith('layers.0.feed_forward.') and
            k.split('.')[3].isdigit() and 'weight' in k))

        emb_dim   = sd['embedding.weight'].shape[0]
        token_dim = sd['embedding.weight'].shape[1]
        num_heads = next(nh for nh in [6, 4, 8, 2, 1] if emb_dim % nh == 0)
        head_dim  = emb_dim // num_heads
        output_dim = sd['output.weight'].shape[0]

        # Calibre inp_ib depuis un batch
        ch_tmp, cn_tmp = next(iter(loader))
        token_tmp = cn_tmp[0].numpy().transpose(1, 2, 0).reshape(cn_tmp.shape[2], -1)
        max_val   = float(np.abs(token_tmp).max()) * 1.1
        inp_ib    = max(int(np.ceil(np.log2(max_val + 1e-8))) + 1, 2)
        print(f"  arch: emb={emb_dim} heads={num_heads} hid={sd[f'layers.0.feed_forward.{ffn_keys[0]}.weight'].shape[0]}"
              f" out={output_dim} seq={cn_tmp.shape[2]} inp_ib={inp_ib}")

        # Charge poids HLS
        W, S_w, B, P, _, _ = load_hls_weights(sd, hw)

        # Modèle PyTorch quantifié
        pt_model, _, _, _ = load_model_quantized(cfg['ckpt'], sd, layer_indices, ffn_keys)

        sr_hls_list = []
        sr_pt_list  = []

        for batch_idx, (ch, ch_norm) in enumerate(loader):
            if batch_idx >= cli.n_batches:
                break

            ch_sq  = ch.squeeze(2)
            ch_np  = ch_norm.numpy()  # (B, 2, 4, 64)
            B_size = ch_np.shape[0]

            # HLS numpy — sample par sample (le HLS traite un sample à la fois)
            sr_hls_batch = []
            for b in range(B_size):
                out_hls = forward_hls_np(
                    ch_np[b], W, S_w, B, P,
                    layer_indices, ffn_keys,
                    num_heads, head_dim, output_dim, inp_ib)
                # out_hls: (OUT//2, S) complex
                norm = torch.linalg.norm(out_hls, keepdim=True)
                out_n = out_hls / (norm + 1e-8)
                # rate_calculator_3d attend (B, ant, users)
                sr = learner.rate_calculator_3d(
                    out_n.unsqueeze(0), ch_sq[b:b+1]).mean().item()
                sr_hls_batch.append(sr)
            sr_hls = float(np.mean(sr_hls_batch))

            # PyTorch quantifié
            with torch.no_grad():
                out_pt  = pt_model(ch_norm.float())
                norm_pt = torch.linalg.norm(out_pt, dim=(1, 2), keepdim=True)
                out_ptn = out_pt / (norm_pt + 1e-8)
                sr_pt   = learner.rate_calculator_3d(out_ptn, ch_sq).mean().item()

            sr_hls_list.append(sr_hls)
            sr_pt_list.append(sr_pt)

            if batch_idx % 5 == 0:
                print(f"  batch {batch_idx:3d}: SR_HLS={sr_hls:.3f}  SR_PT={sr_pt:.3f}"
                      f"  deg={100*(1-sr_hls/sr_pt):.1f}%")

        sr_hls_mean = float(np.mean(sr_hls_list))
        sr_pt_mean  = float(np.mean(sr_pt_list))
        deg         = 100.0 * (1 - sr_hls_mean / sr_pt_mean)

        print(f"\n  ── Résultat {cfg['name']} ──")
        print(f"  SR HLS numpy  : {sr_hls_mean:.4f} bps/Hz  (simule ap_int exact)")
        print(f"  SR PyTorch    : {sr_pt_mean:.4f} bps/Hz  (référence software)")
        print(f"  Dégradation   : {deg:.2f}%")
        if deg < 5:
            print(f"  ✓ HLS validé — dégradation acceptable")
        elif deg < 15:
            print(f"  ⚠ Dégradation modérée — inhérente à la quantification hardware")
        else:
            print(f"  ✗ Dégradation trop élevée — vérifier generate_hls_project.py")

        all_results[cfg['name']] = dict(
            sr_hls=sr_hls_mean, sr_pt=sr_pt_mean, deg=deg)

    print(f"\n{'='*60}")
    print(f"  RÉSUMÉ")
    print(f"{'='*60}")
    for name, r in all_results.items():
        status = '✓' if r['deg'] < 5 else ('⚠' if r['deg'] < 15 else '✗')
        print(f"  {status} [{name}] HLS={r['sr_hls']:.3f}  PT={r['sr_pt']:.3f}"
              f"  deg={r['deg']:.1f}%")

    with open('eval_hls_numpy_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Sauvegardé: eval_hls_numpy_results.json")


def _build_hw_from_sd(sd):
    """Fallback: construit bit_widths depuis le checkpoint si hw_analysis absent."""
    bit_widths = {}
    layer_indices = sorted(set(
        int(k.split('.')[1]) for k in sd
        if k.startswith('layers.') and k.split('.')[1].isdigit()))

    def get_bits(w_key, a_key):
        w_bit = int(np.ceil(float(sd[w_key].float()))) if w_key in sd else 8
        a_bit = int(np.ceil(float(sd[a_key].float()))) if a_key in sd else 8
        return {'weight': w_bit, 'activation': a_bit}

    bit_widths['embedding'] = get_bits(
        'embedding.quan_w_fn.bit', 'embedding.quan_a_fn.bit')
    for i in layer_indices:
        bit_widths[f'layers.{i}.attention'] = get_bits(
            f'layers.{i}.attention.quan_w_fn.bit',
            f'layers.{i}.attention.quan_a_fn.bit')
        ffn_keys = sorted(set(
            int(k.split('.')[3]) for k in sd
            if k.startswith(f'layers.{i}.feed_forward.') and
            k.split('.')[3].isdigit() and 'weight' in k))
        for fk in ffn_keys:
            bit_widths[f'layers.{i}.feed_forward'] = get_bits(
                f'layers.{i}.feed_forward.{fk}.quan_w_fn.bit',
                f'layers.{i}.feed_forward.{fk}.quan_a_fn.bit')
    bit_widths['output'] = get_bits('output.quan_w_fn.bit', 'output.quan_a_fn.bit')
    return {'bit_widths': bit_widths}


if __name__ == '__main__':
    main()