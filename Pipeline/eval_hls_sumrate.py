"""
eval_hls_sumrate.py — Evalue le sum rate de la simulation numpy HLS.
Quantification identique au HLS C++ : poids en ap_fixed depuis les .h generes,
activations via LSQ depuis le checkpoint.
Auto-detecte l architecture depuis le checkpoint.

Usage:
  # Avec hls_dirs explicites (recommande):
  python Pipeline/eval_hls_sumrate.py \
    --configs lambda_1:/path/ckpt_l1 lambda_5:/path/ckpt_l5 \
    --hls_dirs lambda_1:/path/hls_l1 lambda_5:/path/hls_l5 \
    --scenario stecath --n_batches 50

  # Sans hls_dirs: utilise LSQ per-channel (moins fidele au hardware)
  python Pipeline/eval_hls_sumrate.py
"""

import sys, math, json, argparse, re
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
# CLI
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--configs',  nargs='+', default=None,
                   help='name:ckpt_path, ex: lambda_1:/path/ckpt.pth')
    p.add_argument('--hls_dirs', nargs='+', default=None,
                   help='name:hls_dir, ex: lambda_1:/path/hls_weights_l1')
    p.add_argument('--scenario',     default='stecath')
    p.add_argument('--n_batches',    type=int, default=50)
    p.add_argument('--project_root', default=PROJECT_ROOT)
    cli, _ = p.parse_known_args()
    return cli


# ==============================================================================
# Architecture auto-detection
# ==============================================================================

def detect_arch(sd):
    layer_indices = sorted(set(
        int(k.split('.')[1]) for k in sd
        if k.startswith('layers.') and k.split('.')[1].isdigit()
    ))
    num_layers = len(layer_indices)
    emb_dim    = sd['embedding.weight'].shape[0]
    token_dim  = sd['embedding.weight'].shape[1]
    hid_dim    = sd['layers.0.feed_forward.0.weight'].shape[0]
    output_dim = sd['output.weight'].shape[0] if 'output.weight' in sd else emb_dim

    num_heads = None
    try:
        from main_FPGA import TRANSFORMER_CONFIGS
        for cfg_name, cfg_vals in TRANSFORMER_CONFIGS.items():
            if (cfg_vals['embedding_dim'] == emb_dim and
                cfg_vals['hidden_dim']    == hid_dim and
                cfg_vals['num_layers']    == num_layers):
                num_heads = cfg_vals['num_heads']
                break
    except Exception:
        pass
    if num_heads is None:
        for nh in [4, 8, 2, 1]:
            if emb_dim % nh == 0:
                num_heads = nh
                break

    ffn_keys = sorted(set(
        int(k.split('.')[3]) for k in sd
        if k.startswith('layers.0.feed_forward.') and k.split('.')[3].isdigit()
           and 'weight' in k
    ))

    return dict(
        num_layers=num_layers, layer_indices=layer_indices,
        emb_dim=emb_dim, token_dim=token_dim, hid_dim=hid_dim,
        output_dim=output_dim, num_heads=num_heads,
        head_dim=emb_dim // num_heads, ffn_keys=ffn_keys,
    )


def detect_seq_len(scenario, project_root):
    try:
        _saved = sys.argv[:]
        sys.argv = [sys.argv[0], '--scenario', scenario]
        args = input_args().args
        args.scenario = [scenario]
        sys.argv = _saved
        _, loader = utils_(args).Data_Load()
        ch, ch_norm = next(iter(loader))
        return ch_norm.shape[2]
    except Exception:
        return 4


# ==============================================================================
# ap_fixed helpers — identiques a generate_hls_project.py
# ==============================================================================

def read_apfixed_fmt(h_path):
    """Lit format depuis .h genere: cherche 'typedef ap_fixed<TB,IB> w_'"""
    txt = Path(h_path).read_text()
    m = re.search(r'typedef ap_fixed<(\d+),(\d+)>\s+w_', txt)
    if m:
        tb, ib = int(m.group(1)), int(m.group(2))
        return ib, tb - ib  # retourne (int_bits, frac_bits)
    return 8, 8  # fallback


def apfix_np(arr, ib, fb):
    q  = 2**(-fb)
    mv = 2**(ib - 1) - q
    return np.clip(np.round(np.array(arr, dtype=np.float64) / q) * q, -mv, mv)


# ==============================================================================
# LSQ activation helper
# ==============================================================================

def lsq_quantize_act(arr, s, bit):
    b     = max(1, round(float(bit)))
    thd   = 2**b - 1
    s     = abs(float(s))
    inv_s = 1.0 / s if s > 0 else 1.0
    return np.clip(
        np.round(np.array(arr, dtype=np.float64) * inv_s),
        -thd, thd
    ) * s


def rmsnorm_np(x, gamma):
    out = np.zeros_like(x)
    for s in range(x.shape[0]):
        ms = float(np.mean(x[s] ** 2)) + 1e-5
        out[s] = x[s] / np.sqrt(ms) * gamma
    return out


# ==============================================================================
# Charge poids en ap_fixed depuis les .h (mode HLS-fidele)
# ==============================================================================

def load_weights_apfixed(ckpt_path, hls_dir, arch):
    """
    Quantification identique au HLS C++ :
    - Poids : ap_fixed lu depuis les .h generes par generate_hls_project.py
    - Biais : meme format ap_fixed que les poids
    - Gammas RMSNorm : ap_fixed<8,4> fixe
    - Activations LSQ : lus depuis le checkpoint (s, bit)
    """
    hls_dir = Path(hls_dir)
    sd_raw  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd      = sd_raw.get('model_state_dict', sd_raw)
    sd      = {k.replace('module.', ''): v for k, v in sd.items()}
    sd      = {k: v.reshape(1) if v.shape == torch.Size([]) else v
               for k, v in sd.items()}

    def qw(h_name, sd_key):
        h_path = hls_dir / f"{h_name}.h"
        if not h_path.exists() or sd_key not in sd:
            return None
        ib, fb = read_apfixed_fmt(h_path)
        return apfix_np(sd[sd_key].float().numpy(), ib, fb)

    def qb(h_name, sd_key):
        h_path = hls_dir / f"{h_name}.h"
        if not h_path.exists() or sd_key not in sd:
            return None
        ib, fb = read_apfixed_fmt(h_path)
        return apfix_np(sd[sd_key].float().numpy(), ib, fb)

    def qa_params(s_key, bit_key):
        if s_key not in sd:
            return None
        s   = abs(float(sd[s_key].float()))
        bit = float(sd[bit_key].float()) if bit_key in sd else 8.0
        return dict(s=s, bit=bit, thd=2**max(1, round(bit)) - 1)

    w   = {}
    lsq = {}
    fk  = arch['ffn_keys']

    # Embedding
    w['emb_W'] = qw('embedding', 'embedding.weight')
    w['emb_B'] = qb('embedding', 'embedding.bias')
    lsq['emb'] = qa_params('embedding.quan_a_fn.s', 'embedding.quan_a_fn.bit')

    for i in arch['layer_indices']:
        # Attention in_proj
        w[f'ai_{i}_W'] = qw(f'layers_{i}_attention_in_proj',
                             f'layers.{i}.attention.in_proj_weight')
        w[f'ai_{i}_B'] = qb(f'layers_{i}_attention_in_proj',
                             f'layers.{i}.attention.in_proj_bias')
        lsq[f'attn_{i}'] = qa_params(f'layers.{i}.attention.quan_a_fn.s',
                                      f'layers.{i}.attention.quan_a_fn.bit')

        # Attention out_proj — meme format que in_proj
        w[f'ao_{i}_W'] = qw(f'layers_{i}_attention_out_proj',
                             f'layers.{i}.attention.out_proj.weight')
        w[f'ao_{i}_B'] = qb(f'layers_{i}_attention_out_proj',
                             f'layers.{i}.attention.out_proj.bias')

        # FFN
        for fki in fk:
            h_name = f'layers_{i}_feed_forward_{fki}'
            w[f'ff{fki}_{i}_W'] = qw(h_name, f'layers.{i}.feed_forward.{fki}.weight')
            w[f'ff{fki}_{i}_B'] = qb(h_name, f'layers.{i}.feed_forward.{fki}.bias')
            lsq[f'ff{fki}_{i}'] = qa_params(
                f'layers.{i}.feed_forward.{fki}.quan_a_fn.s',
                f'layers.{i}.feed_forward.{fki}.quan_a_fn.bit')

        # RMSNorm gammas — ap_fixed<8,4> fixe comme dans gen_rmsnorm_files
        for ni, nk in [(1, 'norm1'), (2, 'norm2')]:
            key = f'layers.{i}.{nk}.weight'
            if key in sd:
                w[f'norm{ni}_{i}_gamma'] = apfix_np(
                    sd[key].float().numpy(), 4, 4)  # ap_fixed<8,4>

    # Output
    w['out_W'] = qw('output', 'output.weight')
    w['out_B'] = qb('output', 'output.bias')
    lsq['out'] = qa_params('output.quan_a_fn.s', 'output.quan_a_fn.bit')

    # Print summary
    missing = [k for k, v in w.items() if v is None]
    if missing:
        print(f"  [WARN] poids manquants: {missing}")
    print(f"  Poids charges en ap_fixed depuis {hls_dir.name}")
    for k, v in w.items():
        if v is not None and hasattr(v, 'shape'):
            print(f"    {k:20s} shape={v.shape} max={np.abs(v).max():.4f}")
    print(f"  LSQ act params:")
    for k, v in lsq.items():
        if v:
            print(f"    {k:15s} s={v['s']:.6f} bit={v['bit']:.2f} thd=+-{v['thd']}")

    return w, lsq, sd


# ==============================================================================
# Fallback: charge poids en LSQ per-channel (sans hls_dir)
# ==============================================================================

def load_weights_lsq(ckpt_path, arch):
    """Mode fallback sans hls_dir — moins fidele au hardware."""
    sd_raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd     = sd_raw.get('model_state_dict', sd_raw)
    sd     = {k.replace('module.', ''): v for k, v in sd.items()}
    sd     = {k: v.reshape(1) if v.shape == torch.Size([]) else v
              for k, v in sd.items()}

    def np_(key):
        return sd[key].float().numpy() if key in sd else None

    def qw_lsq(w_key, s_key, bit_key):
        w = np_(w_key)
        if w is None: return None
        s   = np_(s_key)
        bit = float(sd[bit_key].float()) if bit_key in sd else 8.0
        if s is None: return w
        if s.ndim == 0: s = s.reshape(1)
        b   = max(1, round(float(bit)))
        thd = 2**b - 1
        s_  = np.abs(s.reshape(-1, 1))
        return np.clip(np.round(w / s_), -thd, thd) * s_

    def qa_params(s_key, bit_key):
        if s_key not in sd: return None
        s   = abs(float(sd[s_key].float()))
        bit = float(sd[bit_key].float()) if bit_key in sd else 8.0
        return dict(s=s, bit=bit, thd=2**max(1, round(bit)) - 1)

    w   = {}
    lsq = {}
    fk  = arch['ffn_keys']

    w['emb_W'] = qw_lsq('embedding.weight', 'embedding.quan_w_fn.s', 'embedding.quan_w_fn.bit')
    w['emb_B'] = np_('embedding.bias')
    lsq['emb'] = qa_params('embedding.quan_a_fn.s', 'embedding.quan_a_fn.bit')

    for i in arch['layer_indices']:
        w[f'ai_{i}_W'] = qw_lsq(f'layers.{i}.attention.in_proj_weight',
                                  f'layers.{i}.attention.quan_w_fn.s',
                                  f'layers.{i}.attention.quan_w_fn.bit')
        w[f'ai_{i}_B'] = np_(f'layers.{i}.attention.in_proj_bias')
        lsq[f'attn_{i}'] = qa_params(f'layers.{i}.attention.quan_a_fn.s',
                                      f'layers.{i}.attention.quan_a_fn.bit')
        w[f'ao_{i}_W'] = qw_lsq(f'layers.{i}.attention.out_proj.weight',
                                  f'layers.{i}.attention.quan_w_out_fn.s',
                                  f'layers.{i}.attention.quan_w_out_fn.bit')
        w[f'ao_{i}_B'] = np_(f'layers.{i}.attention.out_proj.bias')

        for fki in fk:
            pfx = f'layers.{i}.feed_forward.{fki}'
            w[f'ff{fki}_{i}_W'] = qw_lsq(f'{pfx}.weight', f'{pfx}.quan_w_fn.s', f'{pfx}.quan_w_fn.bit')
            w[f'ff{fki}_{i}_B'] = np_(f'{pfx}.bias')
            lsq[f'ff{fki}_{i}'] = qa_params(f'{pfx}.quan_a_fn.s', f'{pfx}.quan_a_fn.bit')

        for ni, nk in [(1, 'norm1'), (2, 'norm2')]:
            g = np_(f'layers.{i}.{nk}.weight')
            w[f'norm{ni}_{i}_gamma'] = g  # FP32 en mode fallback

    w['out_W'] = qw_lsq('output.weight', 'output.quan_w_fn.s', 'output.quan_w_fn.bit')
    w['out_B'] = np_('output.bias')
    lsq['out'] = qa_params('output.quan_a_fn.s', 'output.quan_a_fn.bit')

    print(f"  [WARN] Mode LSQ per-channel (sans hls_dir) — moins fidele au hardware")
    return w, lsq, sd


# ==============================================================================
# Charge modele PyTorch quantifie
# ==============================================================================

def load_pt_model(ckpt_path, arch, project_root):
    sys.path.insert(0, project_root)
    from Pipeline.Transformer_FPGA import StackedTransformer

    sd_raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd     = sd_raw.get('model_state_dict', sd_raw)
    sd     = {k.replace('module.', ''): v for k, v in sd.items()}

    class DotDict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__

    qcfg = DotDict(
        act    = DotDict(mode='lsq', bit=8, per_channel=True,
                         symmetric=False, all_positive=False),
        weight = DotDict(mode='lsq', bit=8, per_channel=True,
                         symmetric=False, all_positive=False),
        excepts= DotDict()
    )
    model = StackedTransformer(
        num_layers=arch['num_layers'], embedding_dim=arch['emb_dim'],
        num_heads=arch['num_heads'], hidden_dim=arch['hid_dim'],
        dropout=0, token_dim=arch['token_dim'],
    )
    rep, _ = quan.find_modules_to_quantize(model, qcfg)
    model  = quan.replace_module_by_names(model, rep)
    sd2    = {k: v.reshape(1) if v.shape == torch.Size([]) else v for k, v in sd.items()}
    model.load_state_dict(sd2, strict=False)
    model.eval()
    return model


# ==============================================================================
# Forward numpy — generique, identique a compute_reference_vectors
# ==============================================================================

def forward_hls_batch(ch_norm_batch, w, lsq, arch, seq_len):
    D   = arch['emb_dim']
    S   = seq_len
    HD  = arch['head_dim']
    NH  = arch['num_heads']
    OUT = arch['output_dim']
    fk  = arch['ffn_keys']
    ff0, ff1 = fk[0], fk[1]

    B          = ch_norm_batch.shape[0]
    results_re = np.zeros((B, OUT // 2, S))
    results_im = np.zeros((B, OUT // 2, S))

    def get(key):
        v = w.get(key)
        return v if v is not None else 0

    def qa(arr, key):
        p = lsq.get(key)
        if p is None:
            return np.array(arr, dtype=np.float64)
        return lsq_quantize_act(arr, p['s'], p['bit'])

    for b in range(B):
        inp      = ch_norm_batch[b]                        # (2, S, ant)
        token_np = inp.transpose(1, 2, 0).reshape(S, -1)  # (S, token_dim)

        # Embedding: LSQ act sur input, puis W_apfixed @ input_lsq + B
        token_q = np.array([qa(token_np[s], 'emb') for s in range(S)])
        B_e = get('emb_B')
        x   = np.array([w['emb_W'] @ token_q[s] + B_e for s in range(S)])

        for i in arch['layer_indices']:
            # Attention
            x_q  = np.array([qa(x[s], f'attn_{i}') for s in range(S)])
            W_ai = w[f'ai_{i}_W']; B_ai = get(f'ai_{i}_B')
            W_ao = w[f'ao_{i}_W']; B_ao = get(f'ao_{i}_B')

            qkv = np.array([W_ai @ x_q[s] + B_ai for s in range(S)])

            ctx = np.zeros((S, D))
            for h in range(NH):
                Q = qkv[:, h*HD:(h+1)*HD]
                K = qkv[:, D+h*HD:D+(h+1)*HD]
                V = qkv[:, 2*D+h*HD:2*D+(h+1)*HD]
                sc = Q @ K.T / math.sqrt(HD)
                sc -= sc.max(axis=1, keepdims=True)
                aw  = np.exp(sc)
                aw /= aw.sum(axis=1, keepdims=True) + 1e-9
                ctx[:, h*HD:(h+1)*HD] = aw @ V

            attn_out = np.array([W_ao @ ctx[s] + B_ao for s in range(S)])
            gamma1   = w[f'norm1_{i}_gamma']
            norm1    = rmsnorm_np(x + attn_out, gamma1)

            # FFN
            norm1_q = np.array([qa(norm1[s], f'ff{ff0}_{i}') for s in range(S)])
            W_f0    = w[f'ff{ff0}_{i}_W']; B_f0 = get(f'ff{ff0}_{i}_B')
            mid     = np.array([
                np.maximum(W_f0 @ norm1_q[s] + B_f0, 0) for s in range(S)])

            mid_q   = np.array([qa(mid[s], f'ff{ff1}_{i}') for s in range(S)])
            W_f1    = w[f'ff{ff1}_{i}_W']; B_f1 = get(f'ff{ff1}_{i}_B')
            ffn_out = np.array([W_f1 @ mid_q[s] + B_f1 for s in range(S)])

            gamma2 = w[f'norm2_{i}_gamma']
            x      = rmsnorm_np(norm1 + ffn_out, gamma2)

        # Output
        x_q  = np.array([qa(x[s], 'out') for s in range(S)])
        W_o  = w['out_W']; B_o = get('out_B')
        raw  = np.array([W_o @ x_q[s] + B_o for s in range(S)])

        half = OUT // 2
        results_re[b] = raw[:, :half].T
        results_im[b] = raw[:, half:].T

    return torch.complex(
        torch.tensor(results_re, dtype=torch.float32),
        torch.tensor(results_im, dtype=torch.float32),
    )


# ==============================================================================
# Loader
# ==============================================================================

def get_loader(scenario, project_root):
    _saved = sys.argv[:]
    sys.argv = [sys.argv[0], '--scenario', scenario]
    args = input_args().args
    args.scenario = [scenario]
    sys.argv = _saved
    _, loader = utils_(args).Data_Load()
    return loader, args


# ==============================================================================
# Evaluation
# ==============================================================================

def evaluate(loader, args, learner, configs, n_batches, arch, seq_len, project_root):
    loaded = {}
    for c in configs:
        print(f"\nChargement {c['name']}...")
        hls_dir = c.get('hls_dir')
        if hls_dir:
            w, lsq, sd = load_weights_apfixed(c['ckpt'], hls_dir, arch)
            mode = 'ap_fixed (HLS-fidele)'
        else:
            w, lsq, sd = load_weights_lsq(c['ckpt'], arch)
            mode = 'LSQ per-channel (fallback)'
        pt_model = load_pt_model(c['ckpt'], arch, project_root)
        loaded[c['name']] = dict(w=w, lsq=lsq, pt=pt_model, mode=mode)
        print(f"  Mode: {mode}")

    results = {c['name']: {'sr_hls': [], 'sr_pt': []} for c in configs}

    print(f"\nEvaluation sur {n_batches} batches (seq_len={seq_len})...")
    for batch_idx, (ch, ch_norm) in enumerate(loader):
        if batch_idx >= n_batches:
            break

        ch_sq = ch.squeeze(2)
        ch_np = ch_norm.numpy()

        for c in configs:
            cfg = loaded[c['name']]

            out_hls   = forward_hls_batch(ch_np, cfg['w'], cfg['lsq'], arch, seq_len)
            norm_hls  = torch.linalg.norm(out_hls, dim=(1, 2), keepdim=True)
            out_hls_n = out_hls / (norm_hls + 1e-8)
            sr_hls    = learner.rate_calculator_3d(out_hls_n, ch_sq).mean().item()
            results[c['name']]['sr_hls'].append(sr_hls)

            with torch.no_grad():
                out_pt    = cfg['pt'](ch_norm.float())
                norm_pt   = torch.linalg.norm(out_pt, dim=(1, 2), keepdim=True)
                out_pt_n  = out_pt / (norm_pt + 1e-8)
                sr_pt     = learner.rate_calculator_3d(out_pt_n, ch_sq).mean().item()
                results[c['name']]['sr_pt'].append(sr_pt)

        if batch_idx % 10 == 0:
            for c in configs:
                r = results[c['name']]
                if r['sr_hls']:
                    print(f"  [{c['name']:12s}] batch {batch_idx:3d}: "
                          f"SR_HLS={r['sr_hls'][-1]:.3f}  SR_PT={r['sr_pt'][-1]:.3f}")

    return results


# ==============================================================================
# Main
# ==============================================================================

def main():
    cli = parse_args()

    # Configs par defaut
    if cli.configs is None:
        base = '/export/tmp/sala/results_fpga/FPGA_small_fpga_stecath/phase2_quant'
        hbase = '/users/sala/Documents/ELE6310/ELE6310E'
        configs_raw  = [
            f"lambda_1:{base}/lambda_1.0/checkpoints/best_model.pth",
            f"lambda_5:{base}/lambda_5.0/checkpoints/best_model.pth",
        ]
        hls_dirs_raw = [
            f"lambda_1:{hbase}/hls_weights_small_fpga_l1",
            f"lambda_5:{hbase}/hls_weights_small_fpga_l5",
        ]
    else:
        configs_raw  = cli.configs
        hls_dirs_raw = cli.hls_dirs or []

    # Parse configs
    configs = []
    for s in configs_raw:
        name, ckpt = s.split(':', 1)
        configs.append(dict(name=name, ckpt=ckpt))

    # Parse hls_dirs
    hls_map = {}
    for s in hls_dirs_raw:
        name, hdir = s.split(':', 1)
        hls_map[name] = hdir
    for c in configs:
        c['hls_dir'] = hls_map.get(c['name'])

    print(f"\n{'='*60}")
    print(f"  eval_hls_sumrate | scenario={cli.scenario} | n_batches={cli.n_batches}")
    print(f"{'='*60}")
    for c in configs:
        mode = 'ap_fixed' if c['hls_dir'] else 'LSQ fallback'
        print(f"  {c['name']:15s} : {c['ckpt']}")
        print(f"  {'':15s}   hls_dir={c['hls_dir']}  [{mode}]")

    # Auto-detect arch
    print(f"\n[1] Detection architecture...")
    sd0_raw = torch.load(configs[0]['ckpt'], map_location='cpu', weights_only=False)
    sd0     = sd0_raw.get('model_state_dict', sd0_raw)
    sd0     = {k.replace('module.', ''): v for k, v in sd0.items()}
    arch    = detect_arch(sd0)
    seq_len = detect_seq_len(cli.scenario, cli.project_root)
    print(f"  {arch}  seq_len={seq_len}")

    print(f"\n[2] Chargement loader...")
    loader, args = get_loader(cli.scenario, cli.project_root)
    learner      = Learner(args)

    print(f"\n[3] Evaluation...")
    results = evaluate(loader, args, learner, configs,
                       cli.n_batches, arch, seq_len, cli.project_root)

    print(f"\n{'='*60}")
    print(f"  Resultats finaux")
    print(f"{'='*60}")
    for name, r in results.items():
        sr_hls = np.mean(r['sr_hls'])
        sr_pt  = np.mean(r['sr_pt'])
        deg    = 100 * (1 - sr_hls / sr_pt) if sr_pt > 0 else float('nan')
        mode   = loaded[name]['mode'] if 'loaded' in dir() else '?'
        print(f"\n  [{name}]")
        print(f"    SR numpy HLS : {sr_hls:.4f} +/- {np.std(r['sr_hls']):.4f} bps/Hz")
        print(f"    SR PyTorch   : {sr_pt:.4f}  +/- {np.std(r['sr_pt']):.4f} bps/Hz")
        print(f"    Degradation  : {deg:.2f}%")
    print(f"{'='*60}")

    # Fix: loaded est dans evaluate, on le reconstruit depuis results
    out = {name: {
        'sr_hls_mean':     float(np.mean(r['sr_hls'])),
        'sr_hls_std':      float(np.std(r['sr_hls'])),
        'sr_pt_mean':      float(np.mean(r['sr_pt'])),
        'sr_pt_std':       float(np.std(r['sr_pt'])),
        'degradation_pct': float(100*(1-np.mean(r['sr_hls'])/np.mean(r['sr_pt']))),
        'arch':            {k: v for k, v in arch.items() if k != 'layer_indices'},
        'seq_len':         seq_len,
        'scenario':        cli.scenario,
    } for name, r in results.items()}

    out_path = f"hls_sumrate_{cli.scenario}.json"
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Resultats sauvegardes dans {out_path}")


if __name__ == '__main__':
    main()