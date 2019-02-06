"""
    Run tests for the PyTorch bindings.

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
import unittest
from typing import Dict
from syrah import File
import numpy as np
from numpy import ndarray
from multiprocessing import Pool


def create_test_data(num_items, num_classes, fixed_len, max_val, min_var_len, max_var_len):
    data_dict = dict()

    with File(syr_path, 'w') as syr:
        for i in range(num_items):
            label = np.random.randint(0, num_classes, size=1, dtype=np.int32)
            fixed_len_array = np.random.random(fixed_len).astype(np.float32)
            var_len_array = np.random.randint(0, max_val,
                                              size=np.random.randint(min_var_len, max_var_len),
                                              dtype=np.int32)
            syr.add_item({
                'label': label,
                'fixed_len_array': fixed_len_array,
                'var_len_array': var_len_array
                })

            item = dict()
            item['label'] = label
            item['fixed_len_array'] = fixed_len_array
            item['var_len_array'] = var_len_array

            data_dict[i] = item
            
    return data_dict


syr_path = '/tmp/syrah_test_data.syr'
num_items = 1_000
num_workers = 8
num_classes = 10
fixed_len = 1_000
max_val = 1_000
min_var_len = 100
max_var_len = 1_000


data_dict = create_test_data(num_items, num_classes, fixed_len, max_val, min_var_len, max_var_len)

syr = File(syr_path, 'r')


def assert_item_read(i):
    item: Dict[str, ndarray] = data_dict[i]
    syr_item: Dict[str, ndarray] = syr.get_item(i)

    for key, value in syr_item.items():
        assert np.all(item[key] == value)


def assert_array_read(i):
    item: Dict[str, ndarray] = data_dict[i]

    for key, value in item.items():
        array = syr.get_array(i, key)
        assert np.all(array == value)


class TestSingleMethods(unittest.TestCase):
    def test_sequential_item_read(self):
        item_idxs = range(num_items)

        for i in item_idxs:
            assert_item_read(i)

    def test_sequential_array_read(self):
        item_idxs = range(num_items)

        for i in item_idxs:
            assert_array_read(i)

    def test_random_item_read(self):
        item_idxs = np.random.permutation(num_items)

        for i in item_idxs:
            assert_item_read(i)

    def test_random_array_read(self):
        item_idxs = np.random.permutation(num_items)
        for i in item_idxs:
            assert_array_read(i)


class TestMultiMethods(unittest.TestCase):
    def test_sequential_item_read(self):
        item_idxs = range(num_items)

        p = Pool(num_workers, initializer=syr.open, initargs=(syr_path, 'r'))
        p.map(assert_item_read, item_idxs)

    def test_sequential_array_read(self):
        item_idxs = range(num_items)

        p = Pool(num_workers, initializer=syr.open, initargs=(syr_path, 'r'))
        p.map(assert_array_read, item_idxs)

    def test_random_item_read(self):
        item_idxs = np.random.permutation(num_items)

        p = Pool(num_workers, initializer=syr.open, initargs=(syr_path, 'r'))
        p.map(assert_item_read, item_idxs)

    def test_random_array_read(self):
        item_idxs = np.random.permutation(num_items)

        p = Pool(num_workers, initializer=syr.open, initargs=(syr_path, 'r'))
        p.map(assert_array_read, item_idxs)


if __name__ == '__main__':
    unittest.main()
