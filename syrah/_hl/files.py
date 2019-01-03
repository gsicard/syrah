"""
    Implements high-level support for file objects.
"""
from typing import Any, Dict, Optional, Type, AnyStr
from types import TracebackType
import numpy as np
from numpy import ndarray
import bson

from .. import version
from .. import config


class File:
    # TODO: add "append" mode.
    # TODO: check that metadata offset + length == file size (both "r" and "w" modes)
    """
        Represent a Syrah dataset file.
    """
    def __init__(self, file_path: Optional[str] = None, mode: Optional[str] = None):
        """
        Create a new file object.
        :param file_path: path to the file on disk
        :param mode: opening mode (currently, only "r" or "w" supported)
        """
        if (file_path is None) != (mode is None):
            raise ValueError(f'file_path and mode need to be both instantiated.')

        self.file_path = None
        self._mode = None
        self._fp = None
        self._version = None
        self._metadata_offset = None
        self._metadata_length = None
        self._metadata = None
        self._item_offset = None

        if file_path is not None:
            self.open(file_path, mode)
            self._init_data()

    def open(self, file_path: str, mode: str):
        self.file_path = file_path
        self._mode = mode

        if self._fp is not None:
            self.close()

        self._fp = open(self.file_path, self._mode + 'b')


        return self

    def _init_data(self):
        """
        Initialize metadata anf file headers.
        :return:
        """
        if self._mode == 'w':
            self._version = version.version
            self._metadata_offset = 0
            self._metadata_length = 0
            self._write_headers()

            self._metadata: Dict[int, Dict[str, Dict[str, Any]]] = dict()
            self._item_offset = config.NUM_BYTES_VERSION + config.NUM_BYTES_METADATA_LENGTH + \
                config.NUM_BYTES_METADATA_LENGTH + config.NUM_BYTES_MAGIC_BYTES
        elif self._mode == 'r':
            self._read_headers()

            self._fp.seek(self._metadata_offset)
            self._metadata = bson.loads(self._fp.read(self._metadata_length))
        else:
            raise ValueError(f'Expected File opening mode to be "r" or "w", got {self._mode}.')

    def __enter__(self):
        """
        Return File object when using a "with" statement.
        :return: File object
        """
        return self

    def __exit__(self, exception_type: Optional[Type[BaseException]], exception_value: Optional[BaseException],
                 traceback: Optional[TracebackType]):
        """
        Explicitly close the File when exiting a "with" context and handle exceptions.
        :param exception_type: type of exception
        :param exception_value: value of exception
        :param traceback: traceback
        :return:
        """
        self.close()

    def close(self):
        """
        Close file object.
        Needs to be called explicitly or use a "with" statement.
        :return:
        """
        if self._fp is None:
            return
        if self._fp.closed:
            return
        if self._mode == 'w':
            self._flush()

        self._fp.close()

    def get_item(self, item: str) -> Dict[str, ndarray]:
        """
        Get an item from the dataset.
        :param item: index of the item in the dataset
        :return: a dictionary of arrays
        """
        if self._fp is None:
            raise IOError('Trying to read an item from a non initialized file.')
        if self._fp.closed:
            raise IOError('Trying to read item from a closed file.')
        if self._mode != 'r':
            raise IOError(f'File is expected to be opened in read mode, got {self._mode}.')
        if item not in self._metadata:
            raise KeyError(f'Item {item} could not be found.')

        item_metadata: Dict[str, Dict[str, Any]] = self._metadata[item]
        data: Dict[str, ndarray] = dict()

        for key, array_metadata in item_metadata.items():
            self._fp.seek(array_metadata['offset'])
            array_serialized: AnyStr = self._fp.read(array_metadata['size'])
            data[key]: ndarray = np.frombuffer(array_serialized, dtype=array_metadata['dtype'])

        return data

    def get_array(self, item: str, key: str) -> ndarray:
        """
        Get an array from the dataset.
        :param item: index of the item in the dataset
        :param key: key of the array to retrieve
        :return: an array
        """
        if self._fp is None:
            raise IOError('Trying to read an array from a non initialized file.')
        if self._fp.closed:
            raise IOError('Trying to read array from a closed file.')
        if self._mode != 'r':
            raise IOError(f'File is expected to be opened in read mode, got {self._mode}.')
        if item not in self._metadata:
            raise KeyError(f'Item {item} could not be found.')
        if key not in self._metadata[item]:
            raise KeyError(f'Key {key} could not be found in item {item}.')

        array_metadata = self._metadata[item][key]

        self._fp.seek(array_metadata['offset'])
        array_serialized = self._fp.read(array_metadata['size'])

        return np.frombuffer(array_serialized, dtype=array_metadata['dtype'])

    def num_items(self) -> int:
        return len(self._metadata)

    def _read_headers(self):
        if self._fp.closed:
            raise IOError('Trying to read headers from a closed file.')
        if self._mode != 'r':
            raise IOError(f'File is expected to be opened in read mode, got {self._mode}.')

        self._fp.seek(0)
        header_version: AnyStr = self._fp.read(config.NUM_BYTES_VERSION)
        header_metadata_offset: AnyStr = self._fp.read(config.NUM_BYTES_METADATA_OFFSET)
        header_metadata_length: AnyStr = self._fp.read(config.NUM_BYTES_METADATA_LENGTH)
        header_magic_bytes = self._fp.read(config.NUM_BYTES_MAGIC_BYTES)

        if header_magic_bytes != config.MAGIC_BYTES:
            raise ValueError(f'Expected magic bytes to be "{config.MAGIC_BYTES}", got "{header_magic_bytes}.')

        self._version = self._version_to_string(header_version)
        self._metadata_offset = np.frombuffer(header_metadata_offset, dtype=np.int64)[0]
        self._metadata_length = np.frombuffer(header_metadata_length, dtype=np.int64)[0]

    def _write_headers(self):
        if self._fp.closed:
            raise IOError('Trying to write headers to a closed file.')
        if self._mode != 'w':
            raise IOError(f'File is expected to be opened in write mode, got {self._mode}.')

        header_version: AnyStr = self._version_to_bytes(self._version)
        header_metadata_offset: AnyStr = np.array([self._metadata_offset], dtype=np.int64).tobytes()
        header_metadata_length: AnyStr = np.array([self._metadata_length], dtype=np.int64).tobytes()

        self._fp.seek(0)
        self._fp.write(header_version)
        self._fp.write(header_metadata_offset)
        self._fp.write(header_metadata_length)
        self._fp.write(config.MAGIC_BYTES)

    def write_item(self, item: str, data: Dict[str, ndarray]):
        """
        Write an item with the given item index to the dataset.
        :param item: index of the item
        :param data: dictionary of arrays
        :return:
        """
        if item in self._metadata:
            raise KeyError(f'Item {item} already in dataset.')

        for key, array in data.items():
            self.write_array(item, key, array)

    def write_array(self, item: str, key: str, array: ndarray):
        """
        Write an array with the given item index and key to the dataset.
        :param item: index of the item
        :param key: key of the array
        :param array: array to write
        :return:
        """
        if self._fp is None:
            raise IOError('Trying to write an item to a non initialized file.')
        if self._fp.closed:
            raise IOError('Trying to write an item to a closed file.')
        if self._mode != 'w':
            raise IOError(f'File is expected to be opened in write mode, got {self._mode}.')
        if type(item) is not str:
            raise ValueError(f'Expected key type to be str, got {type(item)}.')
        if type(key) is not str:
            raise ValueError(f'Expected key type to be str, got {type(key)}.')
        if type(array) is not ndarray:
            raise ValueError(f'Expected value type to be ndarray, got {type(array)}.')

        if item not in self._metadata:
            self._metadata[item] = dict()

        if key in self._metadata[item]:
            raise KeyError(f'Key {key} already in item {item}.')

        array_serialized: AnyStr = array.tobytes()

        self._fp.seek(self._item_offset)
        self._fp.write(array_serialized)

        array_metadata = dict()
        array_metadata['dtype'] = str(array.dtype)
        array_metadata['offset'] = self._item_offset
        array_metadata['size'] = len(array_serialized)

        self._metadata[item][key] = array_metadata
        self._item_offset += len(array_serialized)

    @staticmethod
    def _version_to_string(version_bytes: bytes) -> str:
        """
        Deserialize version number from bytes to string format.
        :param version_bytes: 4-byte array representation of the version number
        :return: string version with format "x.y.z"
        """
        return '.'.join([str(x) for x in np.frombuffer(version_bytes, dtype=np.uint8)[1:]])

    @staticmethod
    def _version_to_bytes(version_string: str) -> bytes:
        """
        Serialize version number from string format.
        :param version_string: string version number of format "x.y.z"
        :return: 4-byte array representation of the version number
        """
        return np.array([0] + [int(s) for s in version_string.split('.')], dtype=np.uint8).tobytes()

    def _flush(self):
        """
        Write headers and metadata to file.
        :return:
        """
        if self._fp.closed:
            raise IOError('Trying to flush to a closed file.')
        if self._mode != 'w':
            raise IOError(f'File is expected to be opened in write mode, got {self._mode}.')

        metadata_serialized: AnyStr = bson.dumps(self._metadata)

        self._metadata_offset = self._item_offset
        self._metadata_length = len(metadata_serialized)
        self._write_headers()

        self._fp.seek(self._item_offset)
        self._fp.write(metadata_serialized)
