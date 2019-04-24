"""
    Implements high-level support for metadata objects.

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
from typing import Union, Dict, AnyStr, Optional, Any, Tuple
from numpy import ndarray

import numpy as np
import bson
import multiprocessing as mp
import ctypes


"""
    Types for the array metadata values:
        - [type] means the metadata is a list of different values of type "type" for each item
        - type  means the metadata is a single value of type "type" for all items
"""
metadata_types = {
    'offset': [np.int64],
    'size': [np.int64],
    'dtype': str
}

metadata_ctypes = {
    'offset': [ctypes.c_int64],
    'size': [ctypes.c_int64],
    'dtype': str
}


class MpNdArray:
    """
    Container for a multiprocessing multi-dimensional array.
    """
    def __init__(self, array: ndarray):
        """
        Create a new multiprocessing multi-dimensional array.
        :param array: array to wrap with multiprocessing container
        """
        self.shape = array.shape
        self.dtype = array.dtype
        self.data = mp.RawArray(self.dtype.char, array.ravel())

    def __getitem__(self, index: Tuple[int, ...]) -> Any:
        """
        Get item from array at the given index
        :param index: index of the item
        :return:
        """
        return self.data[np.ravel_multi_index(index, dims=self.shape)]


class MetadataWriter:
    """
    Represents a metadata writer object.
    """
    def __init__(self):
        """
        Create a new metadata writer object.
        """
        self.metadata: Optional[Dict[AnyStr, Dict[AnyStr, Any]]] = None
        self.data: Optional[ndarray] = None
        self.length: int = 0

    def __len__(self) -> int:
        """
        Return the length of the metadata.
        :return: length of the metadata
        """
        return self.length

    def add_item(self, item_metadata: Dict[AnyStr, Dict[AnyStr, int]]):
        """
        Add a new item to the metadata.
        :param item_metadata: dictionary of array metadata
        :return:
        """
        if self.length == 0:
            self._init_metadata(item_metadata)

        if set(item_metadata.keys()) != set(self.metadata.keys()):
            raise KeyError('Array keys not consistent with previous metadata.')

        self._append_metadata(item_metadata)

        self.length += 1

    def _init_metadata(self, item_metadata: Dict[AnyStr, Dict[AnyStr, int]]):
        self.metadata = dict()
        self.data = []

        for array_index, (array_key, array_metadata) in enumerate(item_metadata.items()):
            if set(array_metadata.keys()) != set(metadata_types.keys()):
                raise KeyError(f'Metadata keys: {", ".join(array_metadata.keys())} do not match:'
                               f' {", ".join(metadata_types.keys())}.')

            self.metadata[array_key] = dict()
            for metadata_index, metadata_key in enumerate(metadata_types.keys()):
                if isinstance(metadata_types[metadata_key], list):
                    self.metadata[array_key][metadata_key] = len(self.data)
                    self.data.append([])
                else:
                    self.metadata[array_key][metadata_key] = array_metadata[metadata_key]

    def _append_metadata(self, item_metadata: Dict[AnyStr, Dict[AnyStr, int]]):
        for array_index, (array_key, array_metadata) in enumerate(item_metadata.items()):
            if set(array_metadata.keys()) != set(metadata_types.keys()):
                raise KeyError(f'Metadata keys: {", ".join(array_metadata.keys())} do not match:'
                               f' {", ".join(metadata_types.keys())}.')

            for metadata_index, metadata_key in enumerate(metadata_types.keys()):
                if isinstance(metadata_types[metadata_key], list):
                    self.data[self.metadata[array_key][metadata_key]].append(array_metadata[metadata_key])
                else:
                    if array_metadata[metadata_key] != self.metadata[array_key][metadata_key]:
                        raise ValueError(f'{metadata_key} value {array_metadata[metadata_key]} not consistent with previous values {self.metadata[array_key][metadata_key]}')

    def tobytes(self) -> AnyStr:
        """
        Serializes the metadata.
        :return: serialized metadata
        """
        metadata_serialized = dict()
        for array_key in self.metadata.keys():
            metadata_serialized[array_key] = dict()
            for metadata_key in metadata_types.keys():
                if isinstance(metadata_types[metadata_key], list):
                    metadata_serialized[array_key][metadata_key] = np.array(self.data[self.metadata[array_key][metadata_key]], dtype=metadata_types[metadata_key][0]).tobytes()
                else:
                    metadata_serialized[array_key][metadata_key] = self.metadata[array_key][metadata_key]

        return bson.dumps(metadata_serialized)


class MetadataReader:
    """
    Represents a metadata reader object.
    """
    def __init__(self, serialized: Optional[AnyStr] = None):
        """
        Create a new metadata reader object and initialize it if necessary.
        :param serialized: serialized metadata for initialization
        """
        self.metadata: Optional[Dict[AnyStr, Dict[AnyStr, Any]]] = None
        self.data: Optional[MpNdArray] = None
        self.length: int = 0

        if serialized is not None:
            self.frombuffer(serialized)

    def __len__(self) -> int:
        """
        Return the length of the metadata.
        :return: length of the metadata
        """
        return self.length

    def frombuffer(self, serialized: AnyStr):
        """
        Deserialize metadata given as an argument.
        :param serialized: serialized metadata
        :return:
        """
        metadata_serialized = bson.loads(serialized)
        self.metadata = dict()
        arrays_list = []

        for array_key in metadata_serialized.keys():
            self.metadata[array_key] = dict()
            for metadata_key in metadata_types.keys():
                if isinstance(metadata_types[metadata_key], list):
                    self.metadata[array_key][metadata_key] = len(arrays_list)
                    arrays_list.append(np.frombuffer(metadata_serialized[array_key][metadata_key],
                                                     dtype=metadata_types[metadata_key][0]))
                else:
                    self.metadata[array_key][metadata_key] = metadata_serialized[array_key][metadata_key]

        self.data = MpNdArray(np.array(arrays_list))
        self.length = len(arrays_list[0])

    def get(self, item: int, array_key: AnyStr, metadata_key: AnyStr) -> Any:
        """
        Get the metadata for the specified item, array and metadata key.
        :param item: index of item
        :param array_key: key of array
        :param metadata_key: key of metadata
        :return: value of the metadata
        """
        if isinstance(metadata_types[metadata_key], list):
            metadata_value = self.data[self.metadata[array_key][metadata_key], item]
        else:
            metadata_value = self.metadata[array_key][metadata_key]

        return metadata_value
