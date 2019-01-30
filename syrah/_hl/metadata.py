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
from typing import List, Dict, AnyStr, Optional, Any
from numpy import ndarray

import numpy as np
import bson

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

# TODO: wrapp arrays with multiprocessing.Array: https://stackoverflow.com/questions/5549190/is-shared-readonly-data-copied-to-different-processes-for-multiprocessing/5550156#5550156
class AbstractMetadata:
    """
        Represent a metadata object.
    """
    def __init__(self):
        """
        Create a new metadata object
        """
        self.metadata: Optional[Dict[AnyStr, Dict[AnyStr, ndarray]]] = None
        self.length: int = 0

    def __len__(self) -> int:
        """
        Return the length of the metadata.
        :return: length of the metadata
        """
        return self.length

    def update_length(self, length: Optional[int] = None):
        """
        Check metadata arrays for proper length and update the internal state.
        :param length: optional length to check
        :return:
        """
        length_initialized = False

        for array_key in self.metadata.keys():
            for metadata_key in metadata_types.keys():
                if isinstance(metadata_types[metadata_key], list):
                    num_items = len(self.metadata[array_key][metadata_key])

                    if not length_initialized:
                        if length is None:
                            length = num_items
                        length_initialized = True
                    if num_items != length:
                        raise ValueError(f'Size of {metadata_key} for array {array_key} do not match expected value:'
                                         f' {num_items} != {length}')

        self.length = length


class MetadataWriter(AbstractMetadata):
    """
    Represent a metadata writer object.
    """
    def add_item(self, item_metadata: Dict[AnyStr, Dict[AnyStr, int]]):
        """
        Add a new item to the metadata.
        :param item_metadata: dictionary of array metadata
        :return:
        """
        # TODO: check metadata types consistency with expected type
        if self.length == 0:
            self.metadata = dict()
            for array_key in item_metadata.keys():
                self.metadata[array_key] = dict()
                for metadata_key in metadata_types.keys():
                    if isinstance(metadata_types[metadata_key], list):
                        self.metadata[array_key][metadata_key] = np.array([], dtype=metadata_types[metadata_key][0])

        if set(item_metadata.keys()) != set(self.metadata.keys()):
            raise KeyError('Array keys not consistent with previous metadata.')

        for array_key, array_metadata in item_metadata.items():
            if set(array_metadata.keys()) != set(metadata_types.keys()):
                raise KeyError(f'Metadata keys: {", ".join(array_metadata.keys())} do not match requirements:'
                               f' {", ".join(metadata_types.keys())}.')

            for metadata_key in metadata_types.keys():
                if isinstance(metadata_types[metadata_key], list):
                    self.metadata[array_key][metadata_key] += np.append(self.metadata[array_key][metadata_key],
                                                                        array_metadata[metadata_key])
                else:
                    self.metadata[array_key][metadata_key] = array_metadata[metadata_key]

        self.length += 1

    def tobytes(self) -> AnyStr:
        """
        Serializes the metadata.
        :return: serialized metadata
        """
        self.update_length(self.length)

        metadata_serialized = dict()
        for array_key in self.metadata.keys():
            metadata_serialized[array_key] = dict()
            for metadata_key in metadata_types.keys():
                if isinstance(metadata_types[metadata_key], list):
                    metadata_serialized[array_key][metadata_key] = self.metadata[array_key][metadata_key].tobytes()
                else:
                    metadata_serialized[array_key][metadata_key] = self.metadata[array_key][metadata_key]

        return bson.dumps(metadata_serialized)


class MetadataReader(AbstractMetadata):
    def __init__(self, serialized: Optional[AnyStr] = None):
        """
        Create a new metadata reader object and initialize it if necessary.
        :param serialized: serialized metadata for initialization
        """
        super().__init__()

        if serialized is not None:
            self.frombuffer(serialized)

    def frombuffer(self, serialized: AnyStr):
        """
        Deserialize metadata given as an argument.
        :param serialized: serialized metadata
        :return:
        """
        metadata_serialized = bson.loads(serialized)
        self.metadata = dict()

        for array_key in metadata_serialized.keys():
            self.metadata[array_key] = dict()
            for metadata_key in metadata_types.keys():
                if isinstance(metadata_types[metadata_key], list):
                    self.metadata[array_key][metadata_key] = np.frombuffer(metadata_serialized[array_key][metadata_key],
                                                                           dtype=metadata_types[metadata_key][0])
                else:
                    self.metadata[array_key][metadata_key] = metadata_serialized[array_key][metadata_key]

        self.update_length()

    def get(self, item: int, array_key: AnyStr, metadata_key: AnyStr) -> Any:
        """
        Get the metadata for the specified item, array and metadata key.
        :param item: index of item
        :param array_key: key of array
        :param metadata_key: key of metadata
        :return: value of the metadata
        """
        if isinstance(metadata_types[metadata_key], list):
            metadata_value = self.metadata[array_key][metadata_key][item]
        else:
            metadata_value = self.metadata[array_key][metadata_key]

        return metadata_value
