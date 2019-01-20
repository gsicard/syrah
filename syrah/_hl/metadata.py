from typing import Union, List, Dict, AnyStr, Optional
from numpy import ndarray

import numpy as np
import bson

metadata_keys = {
    'offset': 0,
    'size': 1,
    'type': 2
}


class MetadataAbstract:
    def __init__(self):
        self.array_keys: Optional[List[AnyStr]] = None
        self.metadata: Optional[Union[List[List[List[int]]], ndarray]] = None
        self.overwrite_array_keys_on_first_item: Optional[bool] = None

    def __len__(self):
        if self.metadata is None:
            return 0
        else:
            return len(self.metadata)


class MetadataWriter(MetadataAbstract):
    def __init__(self, array_keys: Optional[List[AnyStr]] = None, overwrite_array_keys_on_first_item: bool = True):
        super().__init__()

        self.array_keys = dict()
        self.metadata = []
        self.overwrite_array_keys_on_first_item = overwrite_array_keys_on_first_item

        if array_keys is not None:
            self.set_array_keys(array_keys)

    def set_array_keys(self, array_keys: List[AnyStr]):
        self.array_keys = dict()
        for i_key, key in enumerate(array_keys):
            self.array_keys[key] = i_key

    def add_item(self, item_metadata: Dict[AnyStr, Dict[AnyStr, int]]):
        if self.overwrite_array_keys_on_first_item and len(self.metadata) == 0:
            self.set_array_keys(list(item_metadata.keys()))

        for array_key in item_metadata.keys():
            if array_key not in self.array_keys:
                raise KeyError(f'{array_key} not in predefined array keys')
            array_metadata = item_metadata[array_key]

            for metadata_key in metadata_keys:
                if metadata_key not in array_metadata:
                    raise KeyError(f'Unexpected metadata key: {metadata_key}')

        self.metadata.append([[item_metadata[array_key][metadata_key] for metadata_key in metadata_keys]
                              for array_key in self.array_keys])

    def tobytes(self) -> AnyStr:
        full_metadata = dict({
            'keys': self.array_keys,
            'metadata': np.array(self.metadata).tobytes()
        })

        return bson.dumps(full_metadata)


class MetadataReader(MetadataAbstract):
    def __init__(self, serialized: Optional[AnyStr] = None):
        super().__init__()

        self.array_keys = []
        self.metadata = []

        if serialized is not None:
            self.frombuffer(serialized)

    def frombuffer(self, serialized: AnyStr):
        full_metadata = bson.loads(serialized)
        self.array_keys = full_metadata['keys']
        self.metadata = np.frombuffer(full_metadata['metadata'], dtype=np.int64)

    def get(self, item, array_key, metadata_key):
        return self.metadata[item][self.array_keys[array_key]][metadata_keys[metadata_key]]

