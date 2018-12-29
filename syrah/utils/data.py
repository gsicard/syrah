"""
    Dataset utilities for PyTorch.
"""
from typing import List, Tuple
from torch.utils.data.dataset import Dataset
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
        self.syr = File(file_path, 'r')
        self.keys = keys
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
