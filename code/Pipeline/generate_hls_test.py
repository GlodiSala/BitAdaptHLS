"""
generate_hls_test.py -- Genere hls_test_vectors.h
Reference = simulation numpy exacte du HLS :
  - Poids lus depuis les .h (ap_fixed)
  - Activations quantifiees avec LSQ round(bit) lu depuis le .pth

Usage: python Pipeline/generate_hls_test.py
"""
import re, torch, numpy as np, sys, math
sys.path.insert(0, '/users/sala/Documents/ELE6310/ELE6310E')

import quan
from Pipeline.Transformer_FPGA import StackedTransformer
from Pipeline.input_args import input_args
from Pipeline.utils import utils_

CKPT        = '/export/tmp/sala/results_fpga/FPGA_small_fpga_stecath/phase2_quant/lambda_5.0/checkpoints/best_model.pth'
OUTDIR      = '/users/sala/Documents/ELE6310/ELE6310E/'
WEIGHTS_DIR = '/users/sala/Documents/ELE6310/ELE6310E/hls_weights_small_fpga_l5/'

# =============================================================================
# Utilitaires
# =============================================================================

def arr_to_c(name, arr, dtype="float"):
    vals = ", ".join([f"{v:.8f}" for v in arr.flatten()])
    return f"static const {dtype} {name}[{len(arr.flatten())}] = {{{vals}}};\n"

def sep(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

def apfix(arr, ib, fb):
    """Simule ap_fixed<ib+fb,ib> : round + clip"""
    q     = 2.0**(-fb)
    max_v = 2.0**(ib - 1) - q
    return np.clip(np.round(np.array(arr, dtype=np.float64) / q) * q, -max_v, max_v)

def lsq_quant(arr, s, thd_neg, thd_pos):
    """Simule quan_a_fn avec round(bit) — identique au code HLS genere"""
    v = np.round(np.array(arr, dtype=np.float64) / s)
    v = np.clip(v, thd_neg, thd_pos)
    return v * s

def read_apfixed_fmt(layer_key):
    """Lit int_bits et frac_bits depuis le .h genere par extract_weights_hls.py"""
    path = WEIGHTS_DIR + layer_key + '.h'
    txt  = open(path).read()
    m    = re.search(r'typedef ap_fixed<(\d+),(\d+)>\s+w_', txt)
    tb, ib = int(m.group(1)), int(m.group(2))
    return ib, tb - ib

def linear(W, B, x):
    return W @ x + (B if B is not None else 0.0)

def rmsnorm(x, gamma):
    ms = float(np.mean(x**2)) + 1e-5
    return x / np.sqrt(ms) * gamma

# =============================================================================
# Chargement checkpoint — poids bruts + parametres LSQ
# =============================================================================

sep("Chargement checkpoint")
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
sd   = {k.replace('module.',''): v for k,v in ckpt.get('model_state_dict', ckpt).items()}

# Patch scalaires [] -> [1] pour compatibilite
for k in list(sd.keys()):
    if sd[k].shape == torch.Size([]):
        sd[k] = sd[k].reshape(1)

def get_w(sd_key):
    return sd[sd_key].float().numpy()

def qw(layer_key, sd_key):
    """Poids quantifies en ap_fixed — reproduit clip_to_apfixed de extract_weights_hls.py"""
    ib, fb = read_apfixed_fmt(layer_key)
    return apfix(get_w(sd_key), ib, fb)

def qb(layer_key, sd_key):
    """Biais quantifie en ap_fixed"""
    ib, fb = read_apfixed_fmt(layer_key)
    return apfix(get_w(sd_key), ib, fb) if sd_key in sd else None

# =============================================================================
# Collecte parametres LSQ activation depuis le .pth
# Reproduit exactement collect_lsq_act_params() de extract_weights_hls.py
# =============================================================================

sep("Collecte parametres LSQ activation depuis .pth")

class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__

quan_cfg = DotDict(
    act    = DotDict(mode='lsq', bit=8, per_channel=True, symmetric=False, all_positive=False),
    weight = DotDict(mode='lsq', bit=8, per_channel=True, symmetric=False, all_positive=False),
    excepts= DotDict()
)

model = StackedTransformer(num_layers=4, embedding_dim=128, num_heads=4,
                           hidden_dim=512, dropout=0, token_dim=128)
replaced, _ = quan.find_modules_to_quantize(model, quan_cfg)
model = quan.replace_module_by_names(model, replaced)
model.load_state_dict(sd, strict=False)
model.eval()

LSQ_ACT = {}
for name, mod in model.named_modules():
    if hasattr(mod, 'quan_a_fn'):
        qa      = mod.quan_a_fn
        s       = float(qa.s)
        b_r     = round(float(qa.bit))
        thd_neg = -(2**(b_r - 1))
        thd_pos =  (2**(b_r - 1)) - 1
        LSQ_ACT[name] = dict(s=s, bit_r=b_r, thd_neg=thd_neg, thd_pos=thd_pos)
        print(f"  {name:45s} s={s:+.6f} bit_r={b_r} thd=[{thd_neg},{thd_pos}]")

# =============================================================================
# Chargement dataset + input
# =============================================================================

sep("Chargement dataset")
input_args_ = input_args()
arguments   = input_args_.args
arguments.scenario = ['stecath']
_, test_loader = utils_(arguments).Data_Load()
x_batch, _ = next(iter(test_loader))

if x_batch.is_complex():
    x_in = torch.cat([x_batch[:,0:1,:,:].real.float(),
                       x_batch[:,0:1,:,:].imag.float()], dim=1)
else:
    x_in = x_batch.float()

x_tmp = x_in[:1]
size  = x_tmp.size()
print(f"x_tmp : {size}  seq_len={size[2]}")

x_view   = x_tmp.permute(0,2,3,1).contiguous().view(size[0], size[2], size[3]*size[1])
best_tok = int(x_view[0,:,:].abs().max(dim=1).values.argmax())
x_raw    = x_view[0, best_tok, :].numpy()
scale    = 1.0 / float(np.abs(x_raw).max())
print(f"token={best_tok}  scale={scale:.6f}")

emb_ib, emb_fb = read_apfixed_fmt('embedding')
x_q = apfix(x_raw * scale, emb_ib, emb_fb)
print(f"x_q[:4] = {x_q[:4]}")

# =============================================================================
# Simulation HLS numpy
# Reproduit exactement ce que les .cpp generes calculent
# =============================================================================

sep("Simulation HLS (numpy)")

# Embedding
W_e = qw('embedding', 'embedding.weight')
B_e = qb('embedding', 'embedding.bias')
x   = linear(W_e, B_e, x_q)
token0_emb = x.copy()
print(f"emb[:4] = {x[:4]}")

ref_layers = []

for i in range(4):
    print(f"\n  --- Layer {i} ---")

    # Attention : LSQ input → QKV → ctx=V (seq_len=1) → out_proj
    p_ai  = LSQ_ACT[f"layers.{i}.attention"]
    x_lsq = lsq_quant(x, p_ai['s'], p_ai['thd_neg'], p_ai['thd_pos'])

    W_ai = qw(f'layers_{i}_attention_in_proj',  f'layers.{i}.attention.in_proj_weight')
    B_ai = qb(f'layers_{i}_attention_in_proj',  f'layers.{i}.attention.in_proj_bias')
    W_ao = qw(f'layers_{i}_attention_out_proj', f'layers.{i}.attention.out_proj.weight')
    B_ao = qb(f'layers_{i}_attention_out_proj', f'layers.{i}.attention.out_proj.bias')

    qkv      = linear(W_ai, B_ai, x_lsq)
    ctx      = qkv[256:]              # seq_len=1 → softmax=1 → ctx = V
    attn_out = linear(W_ao, B_ao, ctx)
    print(f"  attn_out[:4] = {attn_out[:4]}")

    # Residuelle + RMSNorm1
    g1_ib, g1_fb = 4, 4              # ap_fixed<8,4> hardcode dans gen_rmsnorm_files
    gamma1 = apfix(get_w(f'layers.{i}.norm1.weight'), g1_ib, g1_fb)
    norm1  = rmsnorm(x + attn_out, gamma1)
    print(f"  norm1[:4]    = {norm1[:4]}")

    # FFN : LSQ input → ff0+ReLU → LSQ mid → ff3
    p_f0     = LSQ_ACT[f"layers.{i}.feed_forward.0"]
    x_lsq_f0 = lsq_quant(norm1, p_f0['s'], p_f0['thd_neg'], p_f0['thd_pos'])

    W_f0 = qw(f'layers_{i}_feed_forward_0', f'layers.{i}.feed_forward.0.weight')
    B_f0 = qb(f'layers_{i}_feed_forward_0', f'layers.{i}.feed_forward.0.bias')
    W_f3 = qw(f'layers_{i}_feed_forward_3', f'layers.{i}.feed_forward.3.weight')
    B_f3 = qb(f'layers_{i}_feed_forward_3', f'layers.{i}.feed_forward.3.bias')

    mid      = np.maximum(linear(W_f0, B_f0, x_lsq_f0), 0.0)   # ReLU

    p_f3     = LSQ_ACT[f"layers.{i}.feed_forward.3"]
    mid_lsq  = lsq_quant(mid, p_f3['s'], p_f3['thd_neg'], p_f3['thd_pos'])
    ffn_out  = linear(W_f3, B_f3, mid_lsq)
    print(f"  ffn_out[:4]  = {ffn_out[:4]}")

    # Residuelle + RMSNorm2
    gamma2 = apfix(get_w(f'layers.{i}.norm2.weight'), g1_ib, g1_fb)
    norm2  = rmsnorm(norm1 + ffn_out, gamma2)
    print(f"  norm2[:4]    = {norm2[:4]}")

    ref_layers.append(dict(attn=attn_out, norm1=norm1, ffn=ffn_out, norm2=norm2))
    x = norm2.copy()

# Output : LSQ → linear
p_out    = LSQ_ACT["output"]
x_lsq_o  = lsq_quant(x, p_out['s'], p_out['thd_neg'], p_out['thd_pos'])
W_o      = qw('output', 'output.weight')
B_o      = qb('output', 'output.bias')
raw      = linear(W_o, B_o, x_lsq_o)

out_r   = raw[:64]
out_i   = raw[64:]
out_abs = np.sqrt(out_r**2 + out_i**2)

print(f"\nref_output[:4] = {out_abs[:4]}")
print(f"ref max={out_abs.max():.4f}  min={out_abs.min():.4f}")

# =============================================================================
# Generation hls_test_vectors.h
# =============================================================================

sep("Generation hls_test_vectors.h")

lines = [
    "// Auto-genere par generate_hls_test.py",
    "// Reference = simulation numpy exacte du HLS",
    "// Poids : ap_fixed depuis .h | Activations : LSQ round(bit) depuis .pth",
    f"// token={best_tok}  scale={scale:.6f}",
    "#pragma once\n",
    f"static const float INPUT_SCALE = {scale:.8f};\n",
    "// Input quantifie (128 floats)",
    arr_to_c("test_input", x_q),
    "// Reference embedding (128 floats)",
    arr_to_c("ref_emb", token0_emb),
]

for i, refs in enumerate(ref_layers):
    lines += [
        f"// Layer {i} — attention out", arr_to_c(f"ref_attn_{i}",  refs['attn']),
        f"// Layer {i} — norm1 out",     arr_to_c(f"ref_norm1_{i}", refs['norm1']),
        f"// Layer {i} — ffn out",       arr_to_c(f"ref_ffn_{i}",   refs['ffn']),
        f"// Layer {i} — norm2 out",     arr_to_c(f"ref_norm2_{i}", refs['norm2']),
    ]

lines += [
    "// Reference re",        arr_to_c("ref_re",     out_r),
    "// Reference im",        arr_to_c("ref_im",     out_i),
    "// Reference magnitude", arr_to_c("ref_output", out_abs),
]

out_path = OUTDIR + 'hls_test_vectors.h'
with open(out_path, 'w') as f:
    f.write('\n'.join(lines))

print(f"\nok : {out_path}")
print(f"ref_output[:4] = {out_abs[:4]}")