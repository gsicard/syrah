"""
    Dataset utilities for PyTorch.
"""
from typing import List, Tuple
from torch.utils.data.dataset import Dataset, ConcatDataset
from numpy import ndarray
from .. import File


class SyrahDataset(Dataset):
    """
    Represent a PyTorch Dataset using a Syrah file to store the data.
    """
    def __init__(self, file_path: str, keys: List[str]):
        """
        Create a new `Dataset` object.
        :param file_path: path to the Syrah file
        :param keys: list of keys to retrieve from the data
        """
        self.file_path = file_path
        self.keys = keys

        self.syr = File(file_path, 'r')
        self.len = self.syr.num_items()

    def open(self, worker_id: int = -1):
        """
        Open the dataset file.
        Needs to be set as "worker_init_fn" argument when using a Dataloader with num_workers > 1.
        """
        self.syr.open(self.file_path, 'r')
        self.len = self.syr.num_items()

    def __getitem__(self, item: int) -> Tuple[ndarray]:
        """
        Access the item at the  specified index
        :param item: index of the item
        :return: tuple of arrays
        """
        return tuple([self.syr.get_array(str(item), key) for key in self.keys])

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

    def open(self, worker_id: int = -1):
        """
        Open each of the dataset file.
        Needs to be set as "worker_init_fn" argument when using a Dataloader with num_workers > 1.
        """
        for dataset in self.datasets:
            dataset.open(worker_id)
