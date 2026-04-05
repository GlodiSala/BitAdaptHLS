#!/bin/bash
# =======================================================
# MASTER LAUNCHER FOR grm12 (3x L40S)
# Usage: bash launch_grm12.sh
# =======================================================
set -e

BASE_OUT="/export/tmp/sala/bitadapt_runs"
SCENARIOS=(ericsson stecath)
CFGS=(xlarge large base)
LAMBDAS=(0.1 0.5 1.0 2.0 5.0 10.0 20.0)
FP32_EPOCHS=100
QUANT_EPOCHS=120

mkdir -p logs

echo "============================================"
echo "  MASTER LAUNCHER — grm12"
echo "  $(date)"
echo "============================================"

for CFG in "${CFGS[@]}"; do
    for SCENARIO in "${SCENARIOS[@]}"; do

        RUN_TAG="${SCENARIO}_${CFG}"
        FP32_DIR="${BASE_OUT}/fp32_${RUN_TAG}"
        FP32_PATH="${FP32_DIR}/fp32_${RUN_TAG}.pth"
        mkdir -p "$FP32_DIR"

        # Phase 1 — FP32 pretraining
        FP32_JOB=$(sbatch --parsable \
            --job-name="FP32_${RUN_TAG}" \
            run_job.sh \
                --no-quan \
                --scenario "$SCENARIO" \
                --transformer_cfg "$CFG" \
                --output_dir "$FP32_DIR" \
                --fp32_model_path "$FP32_PATH" \
                --epochs "$FP32_EPOCHS")

        echo "✓ FP32 job submitted: $RUN_TAG → SLURM ID $FP32_JOB"

        # Phase 2 — both pretrained and scratch depend on FP32 finishing
        for LAM in "${LAMBDAS[@]}"; do

            # Pretrained
            OUT_PRE="${BASE_OUT}/quant_pretrained_${RUN_TAG}_lam${LAM}"
            mkdir -p "$OUT_PRE"
            sbatch --parsable \
                --job-name="QP_${RUN_TAG}_l${LAM}" \
                --dependency=afterok:${FP32_JOB} \
                run_job.sh \
                    --quan \
                    --init_mode pretrained \
                    --scenario "$SCENARIO" \
                    --transformer_cfg "$CFG" \
                    --lambda_reg "$LAM" \
                    --output_dir "$OUT_PRE" \
                    --fp32_model_path "$FP32_PATH" \
                    --epochs "$QUANT_EPOCHS" > /dev/null

            # Scratch — waits for FP32 to get normalized loss reference
            OUT_SCR="${BASE_OUT}/quant_scratch_${RUN_TAG}_lam${LAM}"
            mkdir -p "$OUT_SCR"
            sbatch --parsable \
                --job-name="QS_${RUN_TAG}_l${LAM}" \
                --dependency=afterok:${FP32_JOB} \
                run_job.sh \
                    --quan \
                    --init_mode scratch \
                    --scenario "$SCENARIO" \
                    --transformer_cfg "$CFG" \
                    --lambda_reg "$LAM" \
                    --output_dir "$OUT_SCR" \
                    --fp32_model_path "$FP32_PATH" \
                    --epochs "$QUANT_EPOCHS" > /dev/null

        done

        echo "  → $(( ${#LAMBDAS[@]} * 2 )) quant jobs queued for $RUN_TAG"
    done
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u $USER"
echo "============================================"