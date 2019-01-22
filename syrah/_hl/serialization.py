"""
    Implements high-level support for array serialization.

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
from typing import AnyStr, Tuple
from numpy import ndarray
from numpy import dtype

import numpy as np

"""
Supported types by index
"""
dtypes = {
    0: 'bool',
    1: 'int8',
    2: 'uint8',
    3: 'int16',
    4: 'uint16',
    5: 'int32',
    6: 'uint32',
    7: 'int64',
    8: 'uint64',

    23: 'float16',
    11: 'float32',
    12: 'float64',
    13: 'float128',

    19: 'str'
}

"""
Supported types by name
"""
dtype_names = {value: key for key, value in dtypes}


def format_dtype(data_type: dtype) -> str:
    """
    Converts a numpy array type in one of the supported types in string format,
     raises an exception if it is not possible.
    :param data_type: numpy array type
    :return: type in string format
    """
    dtype_num = data_type.num

    if dtype_num not in dtypes:
        raise TypeError(f'Type {data_type} is not supported. Supported types are: {", ".join(dtype_names.keys())}')

    return dtypes[dtype_num]


def deserialize_array(array_serialized: AnyStr, data_type: str) -> ndarray:
    """
    Deserialize an array using the given type.
    :param array_serialized: serialized array as byte string
    :param data_type: type of the resulting array
    :return: numpy array of the specified type
    """
    if data_type not in dtype_names:
        raise TypeError(f'Type {data_type} is not supported. Supported types are: {", ".join(dtype_names.keys())}')

    if data_type == 'str':
        string = array_serialized.decode('utf-8')
        return np.array([string])
    else:
        array = np.frombuffer(array_serialized, dtype=data_type)

    return array


def serialize_array(array: ndarray) -> Tuple[AnyStr, str]:
    """
    Serialize a numpy array to byte string
    :param array: numpy array
    :return: serialized array and corresponding array type
    """
    data_type = format_dtype(array.dtype)

    if data_type == 'str':
        array_serialized = bytes(''.join(array), encoding='utf-8')
    else:
        array_serialized = array.tobytes()

    return array_serialized, data_type
