"""
    Dataset utilities for PyTorch.

    This file is part of Syrah.

    Syrah is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Syrah is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Syrah.  If not, see <https://www.gnu.org/licenses/>.
"""
from typing import List, Tuple, Dict, Callable, Optional
from torch.utils.data.dataset import Dataset, ConcatDataset
from numpy import ndarray
from .. import File


class SyrahDataset(Dataset):
    """
    Represent a PyTorch Dataset using a Syrah file to store the data.
    """
    def __init__(self, file_path: str, keys: List[str], process_funcs: Optional[Dict[str, Callable]] = None):
        """
        Create a new `Dataset` object.
        :param file_path: path to the Syrah file
        :param keys: list of keys to retrieve from the data
        """
        self.file_path = file_path
        self.keys = keys
        self.process_funcs = process_funcs or dict()

        self.syr = File(file_path, 'r')
        self.len = self.syr.num_items()

    def worker_init_fn(self, worker_id: int = -1):
        """
        Open the dataset file.
        Needs to be set as "worker_init_fn" argument when using a Dataloader with num_workers > 1.
        :param worker_id: id of the PyTorch data loader worker calling the method
        """
        self.syr.open(self.file_path, 'r')
        self.len = self.syr.num_items()

    def __getitem__(self, item: int) -> Tuple[ndarray, ...]:
        """
        Access the item at the  specified index
        :param item: index of the item
        :return: tuple of arrays
        """

        processed_arrays = [
            self.process_funcs[key](self.syr.get_array(item, key)) if key in self.process_funcs else self.syr.get_array(item, key)
            for key in self.keys
        ]

        return tuple(processed_arrays)

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        :return: length of the dataset
        """
        return self.len


class SyrahConcatDataset(ConcatDataset):
    """
    Represent a concatenation of syrah datasets.
    """
    def __init__(self, datasets: List[SyrahDataset]):
        """
        Concatenate several syrah datasets
        :param datasets: list of syrah datasets
        """
        super().__init__(datasets)

    def worker_init_fn(self, worker_id: int = -1):
        """
        Open each of the dataset file.
        Needs to be set as "worker_init_fn" argument when using a Dataloader with num_workers > 1.
        :param worker_id: id of the PyTorch data loader worker calling the method
        """
        for dataset in self.datasets:
            dataset.worker_init_fn(worker_id)
