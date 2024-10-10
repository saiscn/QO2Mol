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
from torch_geometric.loader import DataLoader
from .qo2mol import QO2MolDataset

proj_root = str(Path(__file__).parent.parent.parent.resolve())


def prepare_train_dataset(root_dir, split_valid=True):
    tr_dataset = QO2MolDataset(root=root_dir)
    num_samples = len(tr_dataset)
    rand_indices = list(range(num_samples))
    np.random.shuffle(rand_indices)
    num_sep = int(num_samples * 0.9)
    train_split, test_split = rand_indices[:num_sep], rand_indices[num_sep:]

    if split_valid:
        train_num = int(0.8 * num_sep)
        train_idx = train_split[:train_num]
        valid_idx = train_split[train_num:]
        train_dataset = tr_dataset[train_idx]
        valid_dataset = tr_dataset[valid_idx]
        test_dataset = tr_dataset[test_split]
        return train_dataset, valid_dataset, test_dataset
    else:
        return tr_dataset[train_split], tr_dataset[test_split]
