from typing import List, Dict, AnyStr, Optional

import numpy as np
import bson

metadata_types = {
    'offset': [np.int64],
    'size': [np.int64],
    'dtype': str
}


class AbstractMetadata:
    def __init__(self):
        self.metadata: Optional[Dict[AnyStr, Dict[AnyStr, List]]] = None
        self.length: int = 0

    def __len__(self):
        return self.length

    def update_length(self, length: Optional[int] = None):
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
    def add_item(self, item_metadata: Dict[AnyStr, Dict[AnyStr, int]]):
        # TODO: check metadata types consistency with previous item
        if self.length == 0:
            self.metadata = dict()
            for array_key in item_metadata.keys():
                self.metadata[array_key] = dict()
                for metadata_key in metadata_types.keys():
                    if isinstance(metadata_types[metadata_key], list):
                        self.metadata[array_key][metadata_key] = []

        if set(item_metadata.keys()) != set(self.metadata.keys()):
            raise KeyError('Array keys not consistent with previous metadata.')

        for array_key, array_metadata in item_metadata.items():
            if set(array_metadata.keys()) != set(metadata_types.keys()):
                raise KeyError(f'Metadata keys: {", ".join(array_metadata.keys())} do not match requirements:'
                               f' {", ".join(metadata_types.keys())}.')

            for metadata_key in metadata_types.keys():
                if isinstance(metadata_types[metadata_key], list):
                    self.metadata[array_key][metadata_key] += [array_metadata[metadata_key]]
                else:
                    self.metadata[array_key][metadata_key] = array_metadata[metadata_key]

        self.length += 1

    def tobytes(self) -> AnyStr:
        self.update_length(self.length)

        metadata_serialized = dict()
        for array_key in self.metadata.keys():
            metadata_serialized[array_key] = dict()
            for metadata_key in metadata_types.keys():
                if isinstance(metadata_types[metadata_key], list):
                    metadata_serialized[array_key][metadata_key] = np.array(self.metadata[array_key][metadata_key],
                                                                            dtype=metadata_types[metadata_key][0]).tobytes()
                else:
                    metadata_serialized[array_key][metadata_key] = self.metadata[array_key][metadata_key]

        return bson.dumps(metadata_serialized)


class MetadataReader(AbstractMetadata):
    def __init__(self, serialized: Optional[AnyStr] = None):
        super().__init__()

        if serialized is not None:
            self.frombuffer(serialized)

    def frombuffer(self, serialized: AnyStr):
        metadata_serialized = bson.loads(serialized)
        self.metadata = dict()

        for array_key in metadata_serialized.keys():
            self.metadata[array_key] = dict()
            for metadata_key in metadata_types.keys():
                if isinstance(metadata_types[metadata_key], list):
                    self.metadata[array_key][metadata_key] = np.frombuffer(metadata_serialized[array_key][metadata_key],
                                                                           dtype=metadata_types[metadata_key][0]).tolist()
                else:
                    self.metadata[array_key][metadata_key] = metadata_serialized[array_key][metadata_key]

        self.update_length()

    def get(self, item: int, array_key: AnyStr, metadata_key: AnyStr):
        if isinstance(metadata_types[metadata_key], list):
            metadata_value = self.metadata[array_key][metadata_key][item]
        else:
            metadata_value = self.metadata[array_key][metadata_key]

        return metadata_value
