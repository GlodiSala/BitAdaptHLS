import argparse
import logging
import logging.config
import os
import time
import munch
import yaml


def merge_nested_dict(d1, d2):
    # Recursively merges two dictionaries
    for k, v in d2.items():
        if isinstance(v, dict) and k in d1:
            merge_nested_dict(d1[k], v)
        else:
            d1[k] = v
    return d1

def get_config(default_file_path, additional_files=None):
    """
    Loads and merges YAML configuration files.
    additional_files: list of extra yaml paths to merge (optional)
    """
    if not os.path.isfile(default_file_path):
        raise FileNotFoundError(f"Cannot find the default configuration file at {default_file_path}")

    with open(default_file_path) as yaml_file:
        cfg = yaml.safe_load(yaml_file)

    # Only merge additional files if explicitly provided
    if additional_files:
        for f in additional_files:
            if not os.path.isfile(f):
                raise FileNotFoundError(f"Cannot find a configuration file at {f}")
            with open(f) as yaml_file:
                c = yaml.safe_load(yaml_file)
                cfg = merge_nested_dict(cfg, c)

    return munch.munchify(cfg)


def init_logger(experiment_name, output_dir, cfg_file=None):
    time_str = time.strftime("%Y%m%d-%H%M%S")
    exp_full_name = (
        time_str if experiment_name is None else experiment_name + "_" + time_str
    )
    log_dir = output_dir / exp_full_name
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / (exp_full_name + ".log")
    logging.config.fileConfig(cfg_file, defaults={"logfilename": log_file})
    logger = logging.getLogger()
    logger.info("Log file for this run: " + str(log_file))
    return log_dir
