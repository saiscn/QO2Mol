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

from .utils import read_moldata_file, convert_std_input, Timer
from .datasetbase import BaseQMDataset

mp_atomNumber2Idx = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 15: 5, 16: 6, 17: 7, 35: 8, 53: 9}
REF_ENERGIES = [
    -377.1312596544013,
    -23896.71978240409,
    -34329.56617414761,
    -47167.507219805506,
    -62605.90249199728,
    -214201.9540501211,
    -249791.03533796355,
    -288696.57944521913,
    -1615128.7577386901,
    -186866.01419930425,
]


class QO2MolDataset(BaseQMDataset):
    """
    The QO2Mol Dataset

    Example:
        # for main set
        dataset = QO2MolDataset(root=root_dir, scope="main")
        split_indices = dataset.get_split_indices()
        train_split, test_split = split_indices["train"], split_indices["test"]
        train_set = dataset[train_split]
        test_set = dataset[test_split]

        # for test b set
        dataset = QO2MolDataset(root=root_dir, scope="b")
    """

    atom_numbers = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
    train_file_names = [f"QO2Mol_train_chunk_{i}.pkl" for i in range(9)]
    test_file_names = ["QO2Mol_test_a.pkl", "QO2Mol_test_b.pkl"]

    def __init__(self, root, scope="main", force_reload=False, double=False):
        with Timer(f"initinalizing ...", "[Dataset]"):
            self.scope = scope
            self.force_reload = force_reload
            super().__init__(root)

            self.data_dict = torch.load(self.data_filepath)
            self._data, self.slices = self.data_dict["data"], self.data_dict["slices"]
            self.check_double(double=double)

    @property
    def raw_file_names(self):
        return self.train_file_names + self.test_file_names

    @property
    def processed_file_names(self):
        return ["QO2Mol_main_processed.pt", "QO2Mol_b_processed.pt"]

    @property
    def data_filepath(self):
        if self.scope == "main":
            return self.processed_file_names[0]
        elif self.scope == "b":
            return self.processed_file_names[1]

    def process(self):

        # you can use only several specific files by nomination, for example:
        # tr_file_paths = [self.raw_dir/'QO2Mol_train_chunk_0.pkl', self.raw_dir/'QO2Mol_train_chunk_1.pkl']
        tr_file_paths = [self.raw_dir / fn for fn in self.train_file_names]
        te_file_paths = [self.raw_dir / fn for fn in self.test_file_names]

        main_file_paths = tr_file_paths + te_file_paths[0]
        test_b_file = te_file_paths[1]

        # we offer a reference energy table for quick start of usage.
        # you can still employ your own referency energies.
        self._process(main_file_paths, self.processed_paths[0], REF_ENERGIES, True)
        self._process(test_b_file, self.processed_paths[1], REF_ENERGIES)

    def _process(self, file_paths, processed_fp, reference_energies, split=False):
        with Timer("1. Reading mol files..."):
            input_datas, num_per_file = read_moldata_file(file_paths)

        with Timer("2. Converting to tensor..."):
            data_list = convert_std_input(input_datas, mp_atomNumber2Idx, reference_energies)

        with Timer("3. Collating datas..."):
            data, slices = torch_geometric.data.InMemoryDataset.collate(data_list)

        data_dict = {"data": data, "slices": slices}

        if split:
            total_counts = sum(num_per_file, 0)
            te_counts = num_per_file[-1]
            total_indices = list(range(total_counts))
            tr_indices, te_indices = total_indices[:-te_counts], total_indices[-te_counts:]
            data_dict["split_indices"] = {"train": tr_indices, "test": te_indices}

        with Timer(f"4. Saving to: {self.processed_paths[0]}"):
            torch.save(data_dict, self.processed_paths[0])


if __name__ == "__main__":
    proj_root = Path(__file__).resolve().parent.parent.parent.parent
    root_dir = proj_root / "download"
    print(f"data root: {root_dir}")
    dataset = QO2MolDataset(root=root_dir)
    print(dataset)
