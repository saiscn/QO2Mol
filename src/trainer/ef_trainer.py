"""qq:适配energy+force的前馈训练过程"""

import time
from contextlib import suppress
from typing import Iterable, Optional, Sequence, Union
import warnings

import numpy as np
import torch
import torch_geometric

# from timm.utils import accuracy
# from torch_cluster import radius_graph

from src.utils import dist_utils


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AvgBank(object):
    """proxy avgmeters"""

    def __init__(self, sep=", ", verbose=False):
        self.sep = str(sep)
        self.verbose = verbose
        self.avgMeters = dict()
        self.key_order = None
        self._default_key_order = []

    def add(self, key, value, num):
        if key not in self.avgMeters:
            self.avgMeters[key] = AverageMeter()
            self._default_key_order.append(key)  # default: FCFS
        self.avgMeters[key].update(value, num)

    def keys(self):
        return list(self.avgMeters.keys())

    def set_order(self, key_order):
        """allow passing non-existing keys, which would be ignored and not shown in print"""
        if self.verbose:
            for k in key_order:
                if k not in self.avgMeters:
                    warnings.warn(f"[AvgBank] key: {k} not found in avgMeters, would be ignored upon printing.")
        self.key_order = key_order

    def __str__(self):
        ss = ""
        key_order = self.key_order if self.key_order else self._default_key_order
        for key in key_order:
            if key in self.avgMeters:
                ss += f"{key}: {self.avgMeters[key].avg:.4f}{self.sep}"
        return ss

    def toString(self):
        return self.__str__()


class EFPredBank(object):
    """for e&f ddp prediction gather"""

    def __init__(self, distributed=False):
        self.names = []
        self.energy_preds = []
        self.force_preds = []
        self.distributed = distributed

    def gather(self):
        name_list = self.names
        energy_list = torch.cat(self.energy_preds).numpy().tolist()
        force_list = [f.numpy() for f in self.force_preds]  # maybe null
        res = {"name": name_list, "energy": energy_list, "force": force_list}
        if self.distributed:
            outputs = [None for _ in range(dist_utils.get_world_size())]
            torch.distributed.all_gather_object(outputs, res)
            res = {"name": [], "energy": [], "force": []}
            for d in outputs:
                res["name"] += d["name"]
                res["energy"] += d["energy"]
                res["force"] += d["force"]
            # reorganize by name
            sorted_indices = np.argsort(res["name"]).tolist()
            for k in ["name", "energy", "force"]:
                sorted_vs = [res[k][i] for i in sorted_indices]
                res[k] = sorted_vs
        return res

    def update_energy(self, names, energy_preds):
        # TODO: consider that names are not integers, may be string?
        if isinstance(names, (list, tuple)):
            self.names += names
        else:
            self.names += names.detach().cpu().flatten().numpy().tolist()
        self.energy_preds.append(energy_preds.detach().cpu().flatten())

    def update_force(self, batch_force, batch_data):
        """inpt: pred_forces ~ (n, 3)"""
        list_force_preds = self.batch_force_reshape(batch_force, batch_data)
        self.force_preds += list_force_preds

    def batch_force_reshape(self, batch_force, batch_data):
        batch = batch_data.batch  # index
        res = []  # batch_size * (nAtom*3)
        idxes = list(set(batch.cpu().numpy()))
        for idx in idxes:
            mask = batch == idx
            res.append(batch_force[mask].detach().cpu().view(-1))
        return res


def ddp_gather_result(loss_metric, energy_metric, force_metric, distributed, with_force, args, logger):
    optim_args = args["optim"]
    if "eval_energy" in optim_args and "eval_force" in optim_args:
        eval_energy, eval_force = args["optim"]["eval_energy"], args["optim"]["eval_force"]
    else:
        eval_energy, eval_force = 0.8, 0.2
    # handle ddp
    if distributed:
        mae_sum = energy_metric.sum
        mae_cnt = energy_metric.count
        output = [None for _ in range(dist_utils.get_world_size())]
        torch.distributed.all_gather_object(output, (mae_sum, mae_cnt))
        mae_sync = np.array(output).sum(axis=0)  # (wz,2)->(2)
        mae_avg = mae_sync[0] / mae_sync[1]

        if with_force:
            f_mae_sum = force_metric.sum
            f_mae_cnt = force_metric.count
            f_output = [None for _ in range(dist_utils.get_world_size())]
            torch.distributed.all_gather_object(f_output, (f_mae_sum, f_mae_cnt))
            f_mae_sync = np.array(f_output).sum(axis=0)  # (wz,2)->(2)
            f_mae_avg = f_mae_sync[0] / f_mae_sync[1]

            final_mae = eval_energy * mae_avg + eval_force * f_mae_avg
            logger.info(
                f"[Gather]: e_mae:{mae_avg:.5f} f_mae:{f_mae_avg:.5f} final_mae:{final_mae:.5f} loss:{loss_metric.avg:.5f}"
            )

            # TODO loss是否也需要sync?
            return final_mae, loss_metric.avg
    else:
        # non-ddp
        mae_avg = energy_metric.avg
        if with_force:
            f_mae_avg = force_metric.avg
            final_mae = eval_energy * mae_avg + eval_force * f_mae_avg
            logger.info(f"[Gather]: e_mae:{mae_avg:.5f} f_mae:{f_mae_avg:.5f} final_mae:{final_mae:.5f}")
            return final_mae, loss_metric.avg

    logger.info(f"[Gather]: e_mae:{mae_avg:.5f}")
    return mae_avg, loss_metric.avg


def default_forward_fn(model, batch_data):
    z, pos, batch = batch_data["z"], batch_data["pos"], batch_data["batch"]
    out_energy = model(z, pos, batch)
    return out_energy


def batch_forward(model, batch_data, with_force, args):
    forward_fn = args.forward_fn if args.forward_fn is not None else default_forward_fn

    # forward pass.
    z, pos, batch = batch_data["z"], batch_data["pos"], batch_data["batch"]
    if with_force:
        pos.requires_grad_()
        out_energy = forward_fn(model, batch_data)
        if isinstance(out_energy, tuple):
            out_energy, out_forces = out_energy  # adaption for models that output force
        else:
            out_forces = -1 * (
                torch.autograd.grad(
                    out_energy, pos, grad_outputs=torch.ones_like(out_energy), create_graph=True, allow_unused=True
                )[0]
            )
    else:
        out_energy = forward_fn(model, batch_data)

    if out_energy.shape[-1] == 1:
        out_energy = out_energy.view(-1)

    out = {"energy": out_energy}
    if with_force:
        if out_forces.shape[-1] != 3:
            out_forces = out_forces.view(-1, 3)
        out["force"] = out_forces

    return out


def step_one_batch_qmcomp(model, batch_data, loss_fn, norm_factor, with_force, args, logger=None):
    """适配dy获得force的方式"""
    # args
    energy_weight = args["optim"]["energy_coefficient"]
    force_weight = args["optim"]["force_coefficient"]
    task_mean, task_std = norm_factor

    # forward pass.
    out = batch_forward(model, batch_data, with_force, args)

    # allow not to compute loss in validate phrase
    if loss_fn is None:
        return out, None

    # compute loss
    loss = []

    # Energy loss.
    energy_target = batch_data.energy.flatten()
    out_energy = out["energy"]
    energy_loss = loss_fn["energy"](out_energy, (energy_target - task_mean) / task_std)
    energy_loss *= energy_weight
    loss.append(energy_loss)

    # Force loss.
    if with_force:
        out_forces = out["force"]
        force_target = batch_data.force  # (-1, 3)
        force_loss = loss_fn["force"](out_forces, force_target / task_std)
        force_loss *= force_weight
        loss.append(force_loss)

    for lc in loss:
        assert hasattr(lc, "grad_fn")
    loss = sum(loss)
    return out, loss


def train_one_epoch(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    norm_factor: Sequence,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args=None,
    target: Union[int, str, None] = None,
    model_ema=None,
    amp_autocast=suppress,  # suppress=do nothing
    loss_scaler=None,
    clip_grad=None,
    print_freq: int = 100,
    logger=None,
    distributed=False,
    with_force=False,
):
    avgBank = AvgBank(verbose=False)
    start_time = time.perf_counter()
    task_mean, task_std = norm_factor

    for step, batch_data in enumerate(data_loader):
        batch_data = batch_data.to(device)

        # forward
        with amp_autocast():
            out, loss = step_one_batch_qmcomp(model, batch_data, loss_fn, norm_factor, with_force, args, logger)

        # backward
        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(loss, optimizer, parameters=model.parameters())
        else:
            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        # metric
        num_samples = out["energy"].shape[0]
        energy_pred = out["energy"].detach().flatten()
        energy_target = batch_data.energy.flatten()
        energy_err = energy_pred * task_std + task_mean - energy_target
        energy_err = torch.mean(torch.abs(energy_err)).item()

        avgBank.add("loss", loss.item(), num_samples)
        avgBank.add("e_MAE", energy_err, num_samples)

        if with_force:
            force_pred = out["force"].flatten().detach()
            force_target = batch_data.force.flatten()
            force_err = torch.mean(torch.abs(force_pred * task_std - force_target)).item()
            avgBank.add("f_MAE", force_err, num_samples)  # n=force_pred.shape[0]?

        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()

        # print
        if step % print_freq == 0 or step == len(data_loader) - 1:  # time.perf_counter() - wall_print > 15:
            w = time.perf_counter() - start_time
            time_per_step = 1e3 * w / (step + 1)
            header_str = f"Epoch: [{epoch}][{step}/{len(data_loader)}]  cur_loss: {loss:.5f}, "
            avg_str = avgBank.toString()
            time_per_step = 1e3 * w / (step + 1)
            speed_str = f" t/step={time_per_step:.0f}ms"
            lr_str = " lr={:.2e}".format(optimizer.param_groups[0]["lr"])
            info_str = header_str + avg_str + speed_str + lr_str
            logger.info(info_str)

    time_consumed = time.perf_counter() - start_time
    time_str = time.strftime("%Hh %Mmin %Ss", time.gmtime(time_consumed))
    logger.info(f"Train complete in {time_str}.")
    # handle ddp
    loss_metric = avgBank.avgMeters["loss"]
    energy_metric = avgBank.avgMeters["e_MAE"]
    force_metric = avgBank.avgMeters["f_MAE"] if with_force else None
    mae, loss = ddp_gather_result(loss_metric, energy_metric, force_metric, distributed, with_force, args, logger)
    return mae, loss


def evaluate(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    norm_factor: Sequence,
    data_loader: Iterable,
    device: torch.device,
    args: dict = None,
    target: Union[int, str, None] = None,
    amp_autocast=suppress,  # suppress=do nothing,
    print_freq: int = 100,
    logger=None,
    distributed=False,
    with_force=False,
    unit_factor=None,  # unit convert
    return_preds=False,
):
    # check ddp
    distributed = distributed or dist_utils.is_dist_available_and_initialized()

    avgBank = AvgBank(verbose=False)
    start_time = time.perf_counter()
    task_mean, task_std = norm_factor
    predBank = EFPredBank(distributed)

    for step, batch_data in enumerate(data_loader):
        batch_data = batch_data.to(device)

        # forward
        with amp_autocast(), torch.no_grad():
            out, loss = step_one_batch_qmcomp(model, batch_data, loss_fn, norm_factor, with_force, args)

        # metric
        num_samples = out["energy"].shape[0]
        energy_pred = out["energy"].detach().flatten()
        energy_target = batch_data.energy.flatten()
        energy_pred = energy_pred * task_std + task_mean
        energy_pred = energy_pred * unit_factor if unit_factor is not None else energy_pred
        energy_err = energy_pred - energy_target

        energy_err = torch.mean(torch.abs(energy_err)).item()
        avgBank.add("e_MAE", energy_err, num_samples)
        if return_preds:
            names = batch_data["mol_name"]
            predBank.update_energy(names, energy_pred)

        # allow not to compute loss in validate phrase
        if loss is not None:
            avgBank.add("loss", loss.item(), num_samples)

        if with_force:
            force_pred = out["force"].flatten().detach()
            force_target = batch_data.force.flatten()
            force_err = torch.mean(torch.abs(force_pred * task_std - force_target)).item()
            avgBank.add("f_MAE", force_err, num_samples)  # n=force_pred.shape[0]?
            if return_preds:
                predBank.update_force(force_pred, batch_data)

        torch.cuda.synchronize()

        # print
        if step % print_freq == 0 or step == len(data_loader) - 1:
            w = time.perf_counter() - start_time
            speed = w / (step + 1)
            eta = w / (step + 1) * (len(data_loader) - step - 1)
            head_str = f"Evaluate: [{step}/{len(data_loader)}]"
            avg_str = avgBank.toString()
            speed_str = f"time:{w:.3f}s, speed:{speed:.3f}s/step, eta in {eta:.3f}s"
            info_str = head_str + avg_str + speed_str
            logger.info(info_str)

    time_consumed = time.perf_counter() - start_time
    time_str = time.strftime("%Hh %Mmin %Ss", time.gmtime(time_consumed))
    logger.info(f"Evaluate complete in {time_str}.")
    # handle ddp
    loss_metric = avgBank.avgMeters["loss"] if "loss" in avgBank.avgMeters else AverageMeter()
    energy_metric = avgBank.avgMeters["e_MAE"]
    force_metric = avgBank.avgMeters["f_MAE"] if with_force else None
    mae, loss = ddp_gather_result(loss_metric, energy_metric, force_metric, distributed, with_force, args, logger)
    if return_preds:
        preds = predBank.gather()
        return mae, loss, preds
    return mae, loss
