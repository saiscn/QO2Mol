import os

import torch
import torch.distributed as dist


def is_dist_available_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_available_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """minior different from torch.dist
    if not in ddp status, `torch.dist.get_rank` returns -1,
    while we return 0.
    """
    if not is_dist_available_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.local_rank = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        args.rank = 0
        args.local_rank = 0
        return

    args.distributed = True

    torch.cuda.set_device(args.local_rank)
    args.dist_backend = "nccl"
    args.dist_url = "env://"

    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()


def all_reduce(value, device, ReduceOp=dist.ReduceOp.SUM):
    if not torch.is_tensor(value):
        value = torch.Tensor([value]).to(device, non_blocking=True)
    if value.device != device:
        value.device = device
    dist.all_reduce(value, ReduceOp, async_op=False)
    return value.item()


class qBarrier(object):
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, trace_tb):
        if exc_type is not None:
            print(trace_tb)
            raise RuntimeError(exc_value)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()


def ddpCall(fn, /, *args, **kwargs):
    if not is_dist_available_and_initialized():
        return fn(*args, **kwargs)
    rank = get_rank()
    # ddp
    with qBarrier():
        if rank == 0:
            res = fn(*args, **kwargs)
    with qBarrier():
        if rank != 0:
            res = fn(*args, **kwargs)
    return res
