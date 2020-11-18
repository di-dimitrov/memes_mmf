#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.

import lib as sweep
from lib import hyperparam


def get_grid(args):
    max_update = 200000

    return [
        hyperparam("run_type", "train_val"),
        hyperparam("config", "projects/mmf_transformer/configs/localized_narratives/pretrain.yaml"),
        # hyperparam("--fp16", save_dir_key=lambda val: "fp16"),
        hyperparam("training.num_workers", 5),
        hyperparam("dataset", "masked_localized_narratives"),
        hyperparam("model", "mmf_transformer", save_dir_key=lambda val: val),
        # For nlvr2, we are able to fit batch of size 16 on single GPU with 16GB
        # memory. Same number is 32 for VQA2, so scale accordingly
        hyperparam(
            "training.batch_size", [64, 256, 1024, 2048], save_dir_key=lambda val: f"bs{val}"
#            "training.batch_size", [4], save_dir_key=lambda val: f"bs{val}"
        ),
        hyperparam("training.seed", -1, save_dir_key=lambda val: f"s{val}"),
        hyperparam("scheduler.type", ["warmup_cosine"]),
        hyperparam("scheduler.params.num_warmup_steps", 2000),
        hyperparam("scheduler.params.num_training_steps", max_update),
        hyperparam("optimizer.type", "adam_w", save_dir_key=lambda val: val),
        hyperparam(
            "optimizer.params.lr", [5e-5], save_dir_key=lambda val: f"lr{val}"
        ),
        hyperparam("optimizer.params.eps", 1e-8),
        hyperparam(
            "training.max_updates", max_update, save_dir_key=lambda val: f"mu{val}"
        ),
        hyperparam("training.log_format", "json"),
        hyperparam("training.pin_memory", True),
        hyperparam("training.log_interval", 1000),
        hyperparam("training.checkpoint_interval", 1000),
        hyperparam("training.evaluation_interval", 2000),
        hyperparam("training.find_unused_parameters", True),
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)
