import datetime
from pathlib import Path
import re
import torch

from .. import dist_utils


def now_str():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


class qBarrier(object):
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, trace_tb):
        if exc_type is not None:
            if dist_utils.get_rank() == 0:
                print(trace_tb)
            raise RuntimeError(exc_value)

        if dist_utils.is_dist_available_and_initialized():
            torch.distributed.barrier()


def save_ckp(model, optimizer, lr_scheduler=None, save_dir=None, save_file=None, **other_params):
    """save checkpoint on rank 0"""
    # 只在rank 0做保存, 自适应ddpp
    if dist_utils.get_rank() != 0:
        return

    # 路径逻辑，判空 + 合并
    assert not (save_dir is None and save_file is None)
    save_dir = "" if save_dir is None else save_dir
    save_file = f"checkpoint_{now_str()}.pt" if save_file is None else save_file
    save_path = Path(save_dir, save_file)
    print(f"Saving checkpoint to: {save_path} ...")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if lr_scheduler is not None:
        checkpoint["lrscheduler_state_dict"] = lr_scheduler.state_dict()
    checkpoint.update(other_params)
    torch.save(checkpoint, save_path)
    print("Saving checkpoint Done.")


def recover(model: torch.nn.Module, optimizer=None, restore_file=None, strict=True, exclude=[]):
    """recover _summary_

    Parameters
    ----------
    model : _type_
        _description_
    optimizer : _type_, optional
        _description_, by default None
    restore_file : _type_, optional
        _description_, by default None
    strict : bool, optional
        whether to warn mismatch weights, by default True

    Returns
    -------
    dict
        checkpoint dict
    """
    assert restore_file is not None
    if restore_file == "" or not Path(restore_file).is_file():
        # qq: is_file 也校验 exist， 不存在时返回False
        raise FileExistsError(f"file: `{restore_file}`  not exist or is a directory")
    # 恢复
    checkpoint = torch.load(restore_file, map_location=torch.device("cpu"))
    # add ddp - state dict convert
    if dist_utils.is_dist_available_and_initialized():
        k = list(checkpoint["model_state_dict"].keys())[0]
        if not k.startswith("module."):
            # 增加前缀
            model_state_dict = {"module." + k: v for k, v in checkpoint["model_state_dict"].items()}
            checkpoint["model_state_dict"] = model_state_dict
    else:
        k = list(checkpoint["model_state_dict"].keys())[0]
        if k.startswith("module."):
            pattern = "module.([\s\S]*)"
            # 减少前缀
            model_state_dict = {re.findall(pattern, k)[0]: v for k, v in checkpoint["model_state_dict"].items()}
            checkpoint["model_state_dict"] = model_state_dict

    # qq temp to del
    # checkpoint["model_state_dict"].pop('module.atom_embed.atom_type_lin.tp.weight')
    for key in exclude:
        if key in checkpoint["model_state_dict"]:
            del checkpoint["model_state_dict"][key]

    # load considering ddp
    with qBarrier():
        # model
        res = model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        if strict is False and dist_utils.get_rank() == 0:
            if len(res.unexpected_keys) > 0:
                print(
                    f"{len(res.unexpected_keys)} Unexpected key(s) in state_dict: { ','.join(res.unexpected_keys) }. "
                )
            if len(res.missing_keys) > 0:
                print(f"{len(res.missing_keys)} Missing key(s) in state_dict: { ','.join(res.missing_keys) }. ")
        # optimizer
        if optimizer is not None:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                rank = dist_utils.get_rank()
                print(f"rank_{rank}: error occurs when load optimizer state dict.")
                print(repr(e))
        return checkpoint
