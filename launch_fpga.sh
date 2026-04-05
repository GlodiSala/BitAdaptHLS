#!/bin/bash
BASE=/users/sala/Documents/ELE6310/ELE6310E
OUTPUT=/export/tmp/sala/results_fpga
mkdir -p $BASE/logs

for cfg in "small_fpga" "medium_fpga" "tiny_fpga"; do
    RUN_NAME="FPGA_${cfg}_stecath"
    RUN_DIR="$OUTPUT/$RUN_NAME"
    echo "Lancement: $cfg → $RUN_DIR"

    sbatch \
        --job-name=FPGA_${cfg} \
        $BASE/slurm08.sh \
        Pipeline/main_FPGA.py \
        --transformer_cfg $cfg \
        --scenario stecath \
        --pretrain_epochs 100 \
        --quant_epochs 120 \
        --lambda_reg 1.0 2.0 5.0 10.0 20.0 \
        --output_dir $OUTPUT \
        --run_name $RUN_NAME

    echo "  Soumis: $cfg"
    sleep 2
done

echo "Verifie: squeue -u sala"