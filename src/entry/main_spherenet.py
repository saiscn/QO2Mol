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

proj_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(proj_root))

from entry.entry_utils import (
    prepare_args,
    prepare_train_val_dataloader,
    prepare_test_dataloader,
    load_loss,
    load_optim,
    load_lr_scheduler,
)
from src.dataset.load_dataset import prepare_train_dataset
from src.model.spherenet import SphereNet
from src.trainer.ef_trainer import evaluate, train_one_epoch
from src.utils import FileLogger
from qqtools import save_ckp, recover


def freeze_rand(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def prepare_model(args):
    model_cfg = args["model"]
    device = args.device

    model = SphereNet(
        cutoff=5.0,
        num_layers=model_cfg.num_layers,
        hidden_channels=model_cfg.hidden_channels,
        out_channels=1,
        int_emb_size=model_cfg.int_emb_size,
        basis_emb_size_dist=model_cfg.basis_emb_size_dist,
        basis_emb_size_angle=model_cfg.basis_emb_size_angle,
        basis_emb_size_torsion=model_cfg.basis_emb_size_torsion,
        out_emb_channels=model_cfg.out_emb_channels,
        num_spherical=model_cfg.num_spherical,
        num_radial=model_cfg.num_radial,
        envelope_exponent=model_cfg.envelope_exponent,
        num_output_layers=3,
    )

    model = model.to(device)
    return model


def run_training(
    train_loader,
    val_loader,
    test_loader,
    model,
    optimizer,
    lr_scheduler,
    loss_fn,
    norm_factor,
    args,
    logger,
):
    # interject
    device = args["device"]
    log_dir = args["log_dir"]
    clip_grad_norm = args["optim"].get("clip_grad_norm", None)
    max_epochs = args["optim"]["max_epochs"]
    print_freq = args.get("print_freq", 100)
    with_force = args["task"]["regress_force"]

    # init
    start_epoch, val_err, test_err = 0, float("inf"), float("inf")
    best_epoch, best_train_err, best_val_err, best_test_err = 0, float("inf"), float("inf"), float("inf")

    """ init mode & restart mode """
    if (args.init_file is not None) and (args.ckp_file is not None):
        raise ValueError("Cannot use arg.init_file and args.ckp_file simutaneously")

    if args.init_file is not None:
        raise NotImplementedError()

    if args.ckp_file is not None:
        ckp_dict = recover(model, optimizer, restore_file=args.ckp_file)
        start_epoch = ckp_dict["epoch"] + 1
        logger.info(f'after load ckp, lr :{optimizer.param_groups[0]["lr"]}')
        if "lrscheduler_state_dict" not in ckp_dict:
            lr_scheduler.scheduler.last_epoch = ckp_dict["epoch"]
            lr_scheduler.step()

    """ mainstream """
    for epoch in range(start_epoch, max_epochs):
        epoch_start_time = time.perf_counter()

        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        logger.info("training...")
        train_err, train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            loss_fn=loss_fn,
            norm_factor=norm_factor,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            clip_grad=clip_grad_norm,
            logger=logger,
            with_force=with_force,
            print_freq=print_freq,
            args=args,
        )

        if val_loader is not None:
            logger.info("evaluating val...")
            val_err, val_loss = evaluate(
                model,
                data_loader=val_loader,
                loss_fn=None,  # omit the loss
                norm_factor=norm_factor,
                device=device,
                logger=logger,
                with_force=with_force,
                print_freq=print_freq,
                args=args,
            )
            gc.collect()
            torch.cuda.empty_cache()

        if test_loader is not None:
            logger.info("evaluating test...")
            test_err, test_loss = evaluate(
                model,
                data_loader=test_loader,
                loss_fn=None,  # omit the loss
                norm_factor=norm_factor,
                device=device,
                logger=logger,
                with_force=with_force,
                print_freq=print_freq,
                args=args,
            )
            gc.collect()
            torch.cuda.empty_cache()

        if lr_scheduler is not None:
            lr_scheduler.step(metrics=val_err)

        # save every epoch
        save_ckp(
            model,
            optimizer,
            save_dir=log_dir,
            save_file="checkpoint_latest.pt",
            epoch=epoch,
            best_epoch=best_epoch,
            best_train_err=best_train_err,
            best_val_err=best_val_err,
            norm_factor=norm_factor,
            datetime=datetime.datetime.now(),
        )

        # record the best results
        if val_err < best_val_err:
            best_val_err = val_err
            best_test_err = test_err
            best_train_err = train_err
            best_epoch = epoch
            save_ckp(
                model,
                optimizer,
                save_dir=log_dir,  #
                save_file=f"best_valid_checkpoint_ep{epoch}_val{val_err:.3f}.pt",
                epoch=epoch,
                best_epoch=best_epoch,
                best_train_err=best_train_err,
                best_val_err=best_val_err,
                norm_factor=norm_factor,
                elements=args.elements,  # may needed
            )

        info_str = "Epoch: [{epoch}] Target: [{target}] train MAE: {train_mae:.5f}, ".format(
            epoch=epoch, target=args.target, train_mae=train_err
        )
        info_str += "val MAE: {:.5f}, ".format(val_err)
        info_str += "test MAE: {:.5f}, ".format(test_err)
        info_str += "Time: {:.2f}s".format(time.perf_counter() - epoch_start_time)
        logger.info(info_str)

        info_str = "Best -- epoch={}, train MAE: {:.5f}, val MAE: {:.5f}, test MAE: {:.5f}\nLog_dir: {}".format(
            best_epoch, best_train_err, best_val_err, best_test_err, log_dir
        )
        logger.info(info_str)


def main(args):
    """Env"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    """ logger """
    logger = FileLogger(True, True, output_dir=args.log_dir)
    logger.info(args)

    """ Dataset """
    train_dataset, val_dataset, test_dataset = None, None, None
    train_loader, val_loader, test_loader = None, None, None
    train_dataset, val_dataset, test_dataset = prepare_train_dataset(root_dir=proj_root / "download", split_valid=True)
    train_loader, val_loader = prepare_train_val_dataloader(train_dataset, val_dataset, args)

    # use test
    if test_dataset is not None:
        test_loader = prepare_test_dataloader(test_dataset, args)

    # standarize
    norm_factor = [0, 1]
    if args.task.standardize:
        if isinstance(train_dataset, torch.utils.data.Subset):
            norm_factor = train_dataset.dataset.get_norm_factor()
        else:
            norm_factor = train_dataset.get_norm_factor()
    logger.info(f"Use norm_factor: {norm_factor}")

    """ model """
    model = prepare_model(args)
    model = model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Number of params: {}".format(n_parameters))

    """trainer"""
    loss_fn = load_loss(args)
    optimizer = load_optim(args, model)
    lr_scheduler = load_lr_scheduler(optimizer, args)

    run_training(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss_fn=loss_fn,
        norm_factor=norm_factor,
        args=args,
        logger=logger,
    )


if __name__ == "__main__":
    args = prepare_args()
    freeze_rand(args.seed)

    print("====Do TRAIN====")
    main(args)
