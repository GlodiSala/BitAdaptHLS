import logging
from pathlib import Path
import yaml
import argparse
import torch
import numpy as np
import os
import json

import Pipeline.process as process
import util
from Pipeline.input_args import input_args
from Pipeline.Energy import analyze_and_compute_energy
from Pipeline.Learner import Learner
from Pipeline.Transformer import StackedTransformer
from Pipeline.utils import utils_

TRANSFORMER_CONFIGS = {
    'tiny':   dict(num_layers=2, embedding_dim=64,  num_heads=2, hidden_dim=512,  dropout=0.05),
    'small':  dict(num_layers=4, embedding_dim=128, num_heads=4, hidden_dim=1024, dropout=0.05),
    'base':   dict(num_layers=4, embedding_dim=128, num_heads=4, hidden_dim=2048, dropout=0.05),
    'large':  dict(num_layers=6, embedding_dim=256, num_heads=8, hidden_dim=2048, dropout=0.05),
    'xlarge': dict(num_layers=8, embedding_dim=256, num_heads=8, hidden_dim=4096, dropout=0.05),
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario',        type=str,   required=True,
                        help='ericsson or stecath')
    parser.add_argument('--transformer_cfg', type=str,   default='small',
                        help='tiny|small|base|large|xlarge')
    parser.add_argument('--output_dir',      type=str,   required=True,
                        help='Where to save the FP32 model')
    parser.add_argument('--epochs',          type=int,   default=100)
    parser.add_argument('--gpu_id',          type=int,   default=0,
                        help='Which GPU to use')
    return parser.parse_args()

def main():
    cli = parse_args()

    # Setup
    input_args_ = input_args()
    arguments   = input_args_.args
    arguments.scenario = [cli.scenario]

    args = util.get_config("config_quantization.yaml")
    args.device.gpu  = [cli.gpu_id]
    args.device.type = torch.device(f'cuda:{cli.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # Output paths
    out_dir = Path(cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fp32_path      = out_dir / f"fp32_{cli.scenario}_{cli.transformer_cfg}.pth"
    fp32_info_path = out_dir / f"fp32_{cli.scenario}_{cli.transformer_cfg}_info.json"
    log_dir        = util.init_logger(
        f"pretrain_{cli.scenario}_{cli.transformer_cfg}",
        out_dir,
        Path.cwd() / "logging.conf"
    )
    logger = logging.getLogger()

    logger.info(f"Scenario       : {cli.scenario}")
    logger.info(f"Transformer    : {cli.transformer_cfg}")
    logger.info(f"Epochs         : {cli.epochs}")
    logger.info(f"Output         : {fp32_path}")

    # Data
    Func = utils_(arguments)
    train_loader, test_loader = Func.Data_Load()
    Learner_ = Learner(arguments)

    # Model
    tcfg  = TRANSFORMER_CONFIGS[cli.transformer_cfg]
    model = torch.nn.DataParallel(
        StackedTransformer(**tcfg),
        device_ids=[cli.gpu_id]
    ).to(args.device.type)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=0.001, amsgrad=True
    )

    def criterion_fp32(FDP, channel, q=None, m=None, **kwargs):
        return Learner_.criterium_FDP(FDP, channel)

    best_sum_rate = -float('inf')

    for epoch in range(1, cli.epochs + 1):
        loss, _, _      = process.train(
            train_loader, model, criterion_fp32,
            optimizer, None, epoch, [], args, arguments, None
        )
        sum_rate, _, _  = process.validate(
            test_loader, model, Learner_.criterium_FDP,
            epoch, [], args, arguments
        )

        logger.info(f"Epoch {epoch:3d}/{cli.epochs} | Loss: {loss:.4f} | SumRate: {sum_rate:.4f}")

        if sum_rate > best_sum_rate:
            best_sum_rate = sum_rate
            torch.save(model.state_dict(), fp32_path)

            # Count quantizable layers for energy reference
            dummy_Q = [16] * sum(
                1 for m in model.modules()
                if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear,
                                  torch.nn.MultiheadAttention))
            )
            fp32_energy = analyze_and_compute_energy(
                model, Q=dummy_Q, Q_activations=dummy_Q,
                input_size=(1, 64, 128)
            ) * 1e-6

            with open(fp32_info_path, 'w') as f:
                json.dump({
                    "sr":              best_sum_rate,
                    "en":              fp32_energy,
                    "scenario":        cli.scenario,
                    "transformer_cfg": cli.transformer_cfg,
                    "epochs_trained":  epoch
                }, f, indent=4)

            logger.info(f"  ✓ New best saved — SR: {best_sum_rate:.4f} | Energy: {fp32_energy:.4f} µJ")

    logger.info("=" * 60)
    logger.info(f"Training complete. Best SR: {best_sum_rate:.4f}")
    logger.info(f"Model saved to: {fp32_path}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()