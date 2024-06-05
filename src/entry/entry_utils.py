import argparse
import datetime
import gc
import os
import sys
import random
import time
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch
import warnings
import yaml
from torch_geometric.loader import DataLoader

from src.utils.qqtools import qDict
from src.utils.extraloss import L2MAELoss
from src.utils.lr_scheduler import LRScheduler


def load_yaml(path):
    cfg = yaml.load(open(path, "r"), Loader=yaml.UnsafeLoader)
    return cfg


def prepare_args():
    """cmd config"""
    parser = argparse.ArgumentParser("Training networks")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--ckp_file", type=str, default=None)
    parser.add_argument("--test", action="store_true", help="train or test")
    parser.add_argument("--local-rank", type=int, default=None, help="for ddp scheduler compatibility, not actually used")  # fmt: skip
    cmd_args = parser.parse_args()

    # Prioritize using configfile
    base_args = qDict(load_yaml("configs/base.yml"))
    file_args = qDict(load_yaml(cmd_args.config))
    base_args.recursive_update(file_args)
    base_args.ckp_file = cmd_args.ckp_file
    base_args.test = cmd_args.test
    args = base_args
    args.distributed = False
    args.rank = 0

    if args["log_dir"]:
        Path(args["log_dir"]).mkdir(parents=True, exist_ok=True)

    # i/o
    if args.config_file is None and args.ckp_file is None:
        assert args.log_dir is not None
        args.config_file = str(Path(args["log_dir"], "config.yaml"))
        yaml.dump(args.to_dict(), open(args.config_file, "w"))

    return args


def prepare_train_val_dataloader(train_dataset, val_dataset, args):
    """Data Loader"""
    batch_size = args.dataloader.batch_size
    num_workers = args.dataloader.num_workers
    pin_memory = args.dataloader.pin_memory
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        val_loader = None

    return train_loader, val_loader


def prepare_test_dataloader(test_dataset, args):
    batch_size = args.dataloader.batch_size
    num_workers = args.dataloader.num_workers
    pin_memory = args.dataloader.pin_memory
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return test_loader


def load_optim(args, model):
    optimizer = args["optim"].get("optimizer")  # "AdamW"
    optimizer = getattr(torch.optim, optimizer)
    optimizer_params = args["optim"]["optimizer_params"]
    lr = args["optim"]["lr_initial"]

    optimizer = optimizer(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        **optimizer_params,
    )
    return optimizer


def load_loss(args):
    loss_fn = {}
    loss_fn["energy"] = args["optim"].get("loss_energy", "mae")
    loss_fn["force"] = args["optim"].get("loss_force", "l2mae")
    for loss, loss_name in loss_fn.items():
        if loss_name in ["l1", "mae"]:
            loss_fn[loss] = torch.nn.L1Loss()
        elif loss_name == "mse":
            loss_fn[loss] = torch.nn.MSELoss()
        elif loss_name == "l2mae":
            loss_fn[loss] = L2MAELoss()
        else:
            raise NotImplementedError(f"Unknown loss function name: {loss_name}")

    return loss_fn


def load_lr_scheduler(optimizer, args):
    optim_cfg = args["optim"].copy()
    optim_cfg["scheduler_params"]["epochs"] = args["optim"]["max_epochs"]
    optim_cfg["scheduler_params"]["warmup_epochs"] = int(
        optim_cfg["scheduler_params"]["warmup_epochs"] * optim_cfg["scheduler_params"]["epochs"]
    )
    print("load_lr_scheduler max_epochs", optim_cfg["max_epochs"])
    lr_scheduler = LRScheduler(optimizer, optim_cfg)
    return lr_scheduler
