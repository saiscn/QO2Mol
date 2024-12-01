import copy
import glob
import pickle
import re
import time
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.separate import separate
from tqdm import tqdm, trange


class BaseQMDataset(InMemoryDataset):
    def __init__(self, root, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.norm = None
        self.elements = None
        self.split_indices = None
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def check_double(self, double):
        # data is float64 by default when processing
        concerned_keys = ["pos", "energy", "force", "ref_energy", "edge_attr", "edge_d_attr"]
        for key in concerned_keys:
            if key in self._data:
                if not double:
                    self._data[key] = self._data[key].float()
                else:
                    self._data[key] = self._data[key].double()

    def get_norm_factor(self):
        if self.norm is None:
            es = self._data.energy.detach().cpu().numpy()
            _mean, _std = np.mean(es), np.std(es)
            self.norm = (_mean, _std)
        return self.norm

    def get_elements(self):
        """return list of atom numbers in periodic table"""
        if self.elements is None:
            self.elements = sorted(list(set(self._data.z.detach().cpu().numpy().tolist())))
        return self.elements

    def get_split_indices(self):
        if self.split_indices is None:
            self.split_indices = self.data_dict["split_indices"]
        return self.split_indices

    @property
    def raw_dir(self) -> str:
        return Path(self.root, "raw")

    @property
    def processed_dir(self) -> str:
        return Path(self.root, "processed")

    @property
    def raw_file_names(self):
        raise NotImplementedError

    @property
    def processed_file_names(self):
        raise NotImplementedError

    def process(self):
        raise NotImplementedError
