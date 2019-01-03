import unittest
from typing import Dict
from syrah import File
import numpy as np
from numpy import ndarray
from multiprocessing import Pool


class SyrFileTestEnvironment:
    def setUp(self):
        self.syr_path = '/tmp/syrah_test_data.syr'
        self.num_items = 1_000
        self.num_workers = 8
        self.num_classes = 10
        self.fixed_len = 1_000
        self.max_val = 1_000
        self.min_var_len = 100
        self.max_var_len = 1_000

        self.create_test_data()

    def create_test_data(self):
        self.data_dict = dict()

        with File(self.syr_path, 'w') as fp_syr:
            for i in range(self.num_items):
                label = np.random.randint(0, self.num_classes, size=1, dtype=np.int32)
                fixed_len_array = np.random.random(self.fixed_len).astype(np.float32)
                var_len_array = np.random.randint(0, self.max_val,
                                                  size=np.random.randint(self.min_var_len, self.max_var_len),
                                                  dtype=np.int32)
                fp_syr.write_item(str(i), {'label': label,
                                           'fixed_len_array': fixed_len_array,
                                           'var_len_array': var_len_array
                                           })
                item = dict()
                item['label'] = label
                item['fixed_len_array'] = fixed_len_array
                item['var_len_array'] = var_len_array

                self.data_dict[str(i)] = item

    def assert_item_read(self, i):
        item: Dict[str, ndarray] = self.data_dict[str(i)]
        syr_item: Dict[str, ndarray] = self.fp_syr.get_item(str(i))

        for key, value in syr_item.items():
            assert np.all(item[key] == value)

    def assert_array_read(self, i):
        item: Dict[str, ndarray] = self.data_dict[str(i)]

        for key, value in item.items():
            array = self.fp_syr.get_array(str(i), key)
            assert np.all(array == value)


class TestSingleMethods(SyrFileTestEnvironment, unittest.TestCase):
    def test_sequential_item_read(self):
        self.fp_syr = File(self.syr_path, 'r')
        item_idxs = range(self.num_items)

        for i in item_idxs:
            self.assert_item_read(i)

    def test_sequential_array_read(self):
        self.fp_syr = File(self.syr_path, 'r')
        item_idxs = range(self.num_items)

        for i in item_idxs:
            self.assert_array_read(i)

    def test_random_item_read(self):
        self.fp_syr = File(self.syr_path, 'r')
        item_idxs = np.random.permutation(self.num_items)

        for i in item_idxs:
            self.assert_item_read(i)

    def test_random_array_read(self):
        self.fp_syr = File(self.syr_path, 'r')
        item_idxs = np.random.permutation(self.num_items)
        for i in item_idxs:
            self.assert_array_read(i)


class TestMultiMethods(SyrFileTestEnvironment, unittest.TestCase):
    def test_sequential_item_read(self):
        self.fp_syr = File()
        item_idxs = range(self.num_items)

        p = Pool(self.num_workers, initializer=self.fp_syr.open, initargs=(self.syr_path, 'r'))
        p.map(self.assert_item_read, item_idxs)

    # def test_sequential_array_read(self):
    #     self.fp_syr = File()
    #     item_idxs = range(self.num_items)
    #
    #     p = Pool(self.num_workers, initializer=self.fp_syr.open, initargs=(self.syr_path, 'r'))
    #     p.map(self.assert_array_read, item_idxs)
    #
    # def test_random_item_read(self):
    #     self.fp_syr = File()
    #     item_idxs = np.random.permutation(self.num_items)
    #
    #     p = Pool(self.num_workers, initializer=self.fp_syr.open, initargs=(self.syr_path, 'r'))
    #     p.map(self.assert_item_read, item_idxs)
    #
    # def test_random_array_read(self):
    #     self.fp_syr = File()
    #     item_idxs = np.random.permutation(self.num_items)
    #
    #     p = Pool(self.num_workers, initializer=self.fp_syr.open, initargs=(self.syr_path, 'r'))
    #     p.map(self.assert_array_read, item_idxs)


if __name__ == '__main__':
    test = TestSingleMethods()
    test.setUp()
    test.test_sequential_item_read()

    test = TestMultiMethods()
    test.setUp()
    test.test_sequential_item_read()

    # unittest.main()
