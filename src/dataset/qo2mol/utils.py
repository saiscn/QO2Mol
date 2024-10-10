import pickle
import time
import json
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
import torch_geometric
from tqdm import tqdm

from .constant import ensureAtomNumber


class Timer:
    def __init__(self, enter_msg, prefix=None):
        self.enter_msg = enter_msg
        self.prefix = prefix + " " if prefix is not None else str()

    def __enter__(self):
        self.start_time = time.time()
        print(f"{self.prefix}{self.enter_msg}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        end_time = time.time()
        execution_time = end_time - self.start_time
        print(f"{self.prefix}Execution time: {execution_time:.2f} seconds")


def _read_moldata_file(filepath):
    if filepath.suffix == ".pkl":
        mol_datas = pickle.load(open(filepath, "rb"))
    elif filepath.suffix == ".npy":
        mol_datas = np.load(filepath, allow_pickle=True).tolist()
    elif filepath.suffix == ".json":
        mol_datas = json.load(open(filepath, "r"))
    else:
        raise ValueError()
    return mol_datas


def read_moldata_file(filepath):
    if isinstance(filepath, (str, Path)):
        filepaths = [filepath]
    assert isinstance(filepath, (list, tuple))
    datas = [_read_moldata_file(fp) for fp in filepaths]
    num_per_file = [len(ds) for ds in datas]
    datas = sum(datas, [])
    return datas, num_per_file


def convert_std_input(input_datas, mp_atomNumber2Idx, refenergy_weight):
    """
    expected fileds:
            - confid: string.
            - inchikey: string.
            - coordinates: np.array
            - elements: np.array
            - energy: float
            - force: np.array
    Args:
        input_datas (list): _description_
        mp_atomNumber2Idx (dict):
        refenergy_weight (list):

    Returns:
        list: mol datas
    """
    data_list = []
    for i, mol_data in enumerate(tqdm(input_datas)):
        name = mol_data.get("confid", str(i))
        inchi_key = mol_data.get("inchikey", "")
        position = mol_data["coordinates"] if "coordinates" in mol_data else mol_data["position"]
        elements = ensureAtomNumber(mol_data["elements"])
        energy = mol_data["energy"]
        force = mol_data["force"] if "force" in mol_data else np.zeros((1, 3))

        ref_e = np.sum([refenergy_weight[mp_atomNumber2Idx[z]] for z in elements])
        res_e = energy - ref_e

        # convet to torch
        res_e = torch.tensor(res_e, dtype=torch.float64)
        ref_e = torch.tensor(ref_e, dtype=torch.float64)

        force = torch.DoubleTensor(force).view(-1, 3)
        z = torch.tensor(elements, dtype=torch.int64)
        pos = torch.tensor(position, dtype=torch.float64)

        atom_count = z.shape[0]
        _data = torch_geometric.data.Data(
            name=name,
            inchi_key=inchi_key,
            atom_count=atom_count,
            z=z,
            pos=pos,
            energy=res_e,
            ref_energy=ref_e,
            force=force,
        )
        data_list.append(_data)
    return data_list
