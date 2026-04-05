# BitAdapt — Mixed-Precision MIMO Precoder on FPGA


## Key Results (medium_fpga, stecath dataset, Xilinx Alveo U280)
| Configuration | SR (bps/Hz) | LUT | DSP | BRAM |
|---|---|---|---|---|
| FP32 reference | 24.52 | — | — | — |
| BitAdapt λ=5 (SW) | 23.06 | — | — | — |
| BitAdapt λ=5 (HLS) | 21.07 | 7% | 1% | 11% |

## Reproduce
```bash
conda env create -f environment.yml
conda activate bitadapt

# Generate HLS project
python code/Pipeline/generate_hls_project.py \
  --checkpoint checkpoints/medium_fpga_fp32_best.pth \
  --hw_analysis hls_analysis/hw_analysis_lambda_5.0.json \
  --output_dir ./hls_output --scenario stecath

# Evaluate HLS SR
python code/Pipeline/eval_delayed_scale.py \
  --scenario stecath --n_batches 30 \
  --configs lambda_5:checkpoints/medium_fpga_fp32_best.pth
```
