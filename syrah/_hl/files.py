"""
    Implements high-level support for file objects.
    
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
from typing import Union, Dict, Optional, Type
from types import TracebackType

from .metadata import MetadataReader, MetadataWriter
from .serialization import *
from .. import version
from .. import config


class File:
    """
        Represents a Syrah dataset file.
    """
    def __init__(self, file_path: str, mode: str):
        """
        Create a new file object.
        :param file_path: path to the file on disk
        :param mode: opening mode (currently, only "r" or "w" supported)
        """
        self._file_path = file_path
        self._mode = mode
        self._fp = None
        self._version = None
        self._metadata_offset = None
        self._metadata_length = None
        self._metadata: Optional[Union[MetadataReader, MetadataWriter]] = None
        self._item_offset = None

        self.open(file_path, mode)
        self._init_data()

    def open(self, file_path: str, mode: str):
        """
        Open the dataset file.
        This method needs to be called by each worker in case of multiprocessing to avoid concurrency issues.
        :param file_path: path to the file on disk
        :param mode: opening mode (currently, only "r" or "w" supported)
        :return:
        """
        self._file_path = file_path
        self._mode = mode

        if self._fp is not None:
            self.close()

        self._fp = open(self._file_path, self._mode + 'b')

    def _init_data(self):
        """
        Initialize metadata and file headers.
        :return:
        """
        if self._mode == 'w':
            self._init_data_write()
        elif self._mode == 'r':
            self._init_data_read()
        else:
            raise ValueError(f'Expected File opening mode to be "r" or "w", got {self._mode}.')

    def _init_data_write(self):
        self._version = version.version

        self._metadata = MetadataWriter()
        self._item_offset = config.NUM_BYTES_VERSION + config.NUM_BYTES_METADATA_LENGTH + \
            config.NUM_BYTES_METADATA_LENGTH + config.NUM_BYTES_MAGIC_BYTES

    def _init_data_read(self):
        self._read_headers()

        self._fp.seek(self._metadata_offset)
        self._metadata = MetadataReader(self._fp.read(self._metadata_length))

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

    def validate_file_handle(self, mode):
        if mode == 'r':
            message = 'Trying to read an item from'
        elif mode == 'w':
            message = 'Trying to write an item to'
        else:
            raise ValueError(f'Unknown mode "{mode}"')

        if self._fp is None:
            raise IOError(f'{message} a non initialized file.')
        if self._fp.closed:
            raise IOError(f'{message} a closed file.')
        if self._mode != mode:
            raise IOError(f'File is expected to be opened in "{mode}" mode, got "{self._mode}".')

    def get_item(self, index: int) -> Dict[str, ndarray]:
        """
        Get an item from the dataset.
        :param index: index of the item in the dataset
        :return: a dictionary of (array name, array data) pairs
        """
        self.validate_file_handle('r')

        if index >= len(self._metadata):
            raise IndexError(f'Item {index} out of range.')

        data: Dict[str, ndarray] = dict()

        for key, array_metadata in self._metadata.metadata.items():
            self._fp.seek(self._metadata.get(index, key, 'offset'))
            array_serialized: AnyStr = self._fp.read(self._metadata.get(index, key, 'size'))

            data[key]: ndarray = deserialize_array(array_serialized, self._metadata.get(index, key, 'dtype'))

        return data

    def get_array(self, index: int, array_name: str, dtype: Optional[str] = None) -> ndarray:
        """
        Get an array from the dataset.
        :param index: index of the item in the dataset
        :param array_name: name of the array to retrieve
        :param dtype: type of the array to decode (if specified, overrides the value contained in the metadata)
        :return: an array
        """
        self.validate_file_handle('r')

        if index >= len(self._metadata):
            raise IndexError(f'Item {index} out of range.')

        if array_name not in self._metadata.metadata:
            raise KeyError(f'Key {array_name} could not be found in metadata.')

        self._fp.seek(self._metadata.get(index, array_name, 'offset'))
        array_serialized = self._fp.read(self._metadata.get(index, array_name, 'size'))
        dtype = dtype or self._metadata.get(index, array_name, 'dtype')

        return deserialize_array(array_serialized, dtype)

    def num_items(self) -> int:
        """
        Get the number of items in the dataset.
        :return: number of items
        """
        return len(self._metadata)

    def _read_headers(self):
        """
        Read the headers and the metadata from file.
        :return:
        """
        self.validate_file_handle('r')

        self._fp.seek(0)
        header_magic_bytes = self._fp.read(config.NUM_BYTES_MAGIC_BYTES)

        if header_magic_bytes != config.MAGIC_BYTES:
            raise ValueError(f'Expected magic bytes to be "{config.MAGIC_BYTES}", got "{header_magic_bytes}.')

        header_version: AnyStr = self._fp.read(config.NUM_BYTES_VERSION)
        header_metadata_offset: AnyStr = self._fp.read(config.NUM_BYTES_METADATA_OFFSET)
        header_metadata_length: AnyStr = self._fp.read(config.NUM_BYTES_METADATA_LENGTH)

        self._version = self._version_to_string(header_version)
        self._metadata_offset = np.frombuffer(header_metadata_offset, dtype=np.int64)[0]
        self._metadata_length = np.frombuffer(header_metadata_length, dtype=np.int64)[0]

    def _write_headers(self):
        """
        Write headers to file.
        :return:
        """
        self.validate_file_handle('w')

        header_version: AnyStr = self._version_to_bytes(self._version)
        header_metadata_offset: AnyStr = np.array([self._metadata_offset], dtype=np.int64).tobytes()
        header_metadata_length: AnyStr = np.array([self._metadata_length], dtype=np.int64).tobytes()

        self._fp.seek(0)
        self._fp.write(config.MAGIC_BYTES)
        self._fp.write(header_version)
        self._fp.write(header_metadata_offset)
        self._fp.write(header_metadata_length)

    @staticmethod
    def create_array_metadata(offset: int, size: int, dtype: str):
        """
        Create metadata dictionary
        :param offset: array offset in the file
        :param size: size of the serialized array
        :param dtype: type of the array data
        :return:
        """
        array_metadata = dict()
        array_metadata['offset'] = offset
        array_metadata['size'] = size
        array_metadata['dtype'] = dtype

        return array_metadata

    def add_item(self, item: Dict[str, ndarray]):
        """
        Write an item with the given index to the dataset.
        :param item: dictionary of (array name, array data) pairs
        :return:
        """
        self.validate_file_handle('w')

        item_offset = self._item_offset
        metadata_item = dict()

        for array_name, array_value in item.items():
            if type(array_name) is not str:
                raise ValueError(f'Expected array_name type to be str, got {type(array_name)}.')
            if type(array_value) is not ndarray:
                raise ValueError(f'Expected value type to be ndarray, got {type(array_value)}.')

            array_serialized, dtype = serialize_array(array_value)

            self._fp.seek(item_offset)
            self._fp.write(array_serialized)

            array_metadata = self.create_array_metadata(item_offset, len(array_serialized), dtype)

            metadata_item[array_name] = array_metadata
            item_offset += len(array_serialized)

        self._metadata.add_item(metadata_item)
        self._item_offset = item_offset

    @staticmethod
    def _version_to_string(version_bytes: bytes) -> str:
        """
        Deserialize version number from little endian encoded bytes to string format.
        :param version_bytes: 4-byte array representation of the version number
        :return: string version with format "x.y.z"
        """
        return '.'.join([str(x) for x in np.frombuffer(version_bytes, dtype=np.uint8)[1:]])

    @staticmethod
    def _version_to_bytes(version_string: str) -> bytes:
        """
        Serialize version number from string format to little endian encoded bytes.
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

        metadata_serialized: AnyStr = self._metadata.tobytes()

        self._metadata_offset = self._item_offset
        self._metadata_length = len(metadata_serialized)
        self._write_headers()

        self._fp.seek(self._item_offset)
        self._fp.write(metadata_serialized)
