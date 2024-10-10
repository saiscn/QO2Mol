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
    """

    atom_numbers = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
    main_file_names = [f"QO2Mol_main_chunk_{i}.pkl" for i in range(10)]
    other_file_names = ["QO2Mol_set_b.pkl", "QO2Mol_set_c.pkl"]

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

        main_file_paths = [self.raw_dir / fn for fn in self.main_file_names]
        set_b_file = self.raw_dir / self.other_file_names[0]

        # we offer a reference energy table for quick start of usage.
        # you can still employ your own referency energies.
        self._process(main_file_paths, self.processed_paths[0], REF_ENERGIES, True)
        self._process(set_b_file, self.processed_paths[1], REF_ENERGIES)

    def _process(self, file_paths, processed_fp, reference_energies):
        with Timer("1. Reading mol files..."):
            input_datas, num_per_file = read_moldata_file(file_paths)

        with Timer("2. Converting to tensor..."):
            data_list = convert_std_input(input_datas, mp_atomNumber2Idx, reference_energies)

        with Timer("3. Collating datas..."):
            data, slices = torch_geometric.data.InMemoryDataset.collate(data_list)

        data_dict = {"data": data, "slices": slices}

        with Timer(f"4. Saving to: {processed_fp}"):
            torch.save(data_dict, processed_fp)


if __name__ == "__main__":
    proj_root = Path(__file__).resolve().parent.parent.parent.parent
    root_dir = proj_root / "download"
    print(f"data root: {root_dir}")
    dataset = QO2MolDataset(root=root_dir)
    print(dataset)
