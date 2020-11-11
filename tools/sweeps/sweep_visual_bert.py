#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.

import lib as sweep
from lib import hyperparam


def get_grid(args):
    max_update = 200_000

    return [
        hyperparam("run_type", "test"),
        hyperparam("config", "projects/mmf_transformer/configs/vqa2/defaults.yaml"),
        # hyperparam("--fp16", save_dir_key=lambda val: "fp16"),
        hyperparam("training.num_workers", 0),
        hyperparam("dataset", "vqa2"),
        hyperparam("model", "mmf_transformer", save_dir_key=lambda val: val),
        # For nlvr2, we are able to fit batch of size 16 on single GPU with 16GB
        # memory. Same number is 32 for VQA2, so scale accordingly
        hyperparam(
            "training.batch_size", [64], save_dir_key=lambda val: f"bs{val}"
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
        hyperparam("checkpoint.max_to_keep", 0),
        hyperparam("checkpoint.resume_file", "/checkpoint/sash/jobs/mmf/checkpoints/mmf_transformer/11_10_2020_vqa2_from_scratch/vqa_scratch_bs64.mmf_transformer.bs64.s-1.adam_w.lr5e-05.mu200000.ngpu8/best.ckpt"),
        hyperparam("evaluation.predict", True)
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)
