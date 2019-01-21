"""
    Run tests for the high level interface

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
from syrah import File
from syrah.utils.data import SyrahDataset, SyrahConcatDataset
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
import pickle


BASE_PATH = '/tmp/syrah_test_data'
NUM_ITEMS = 1_000
MAX_ITEMS = 500
NUM_WORKERS = 8
BATCH_SIZE = 1
NUM_CLASSES = 10
FIXED_LEN = 1_000
MAX_VAL = 1_000
MIN_VAR_LEN = 100
MAX_VAR_LEN = 1_000


def create_test_data(base_path, num_items, max_items):
    data_dict = dict()
    fp_syr = None
    i_syr = 0
    i_local = 0

    for i in range(num_items):
        if i % max_items == 0:
            if fp_syr is not None:
                fp_syr.close()
            fp_syr = File(base_path + '_' + str(i_syr) + '.syr', 'w')
            i_syr += 1
            i_local = 0

        label = np.random.randint(0, NUM_CLASSES, size=1, dtype=np.int32)
        fixed_len_array = np.random.random(FIXED_LEN).astype(np.float32)
        var_len_array = np.random.randint(0, MAX_VAL, size=np.random.randint(MIN_VAR_LEN, MAX_VAR_LEN), dtype=np.int32)

        fp_syr.add_item({
            'idx': np.array([i], dtype=np.int32),
            'label': label,
            'fixed_len_array': fixed_len_array,
            'var_len_array': var_len_array
        })
        pickled_item = dict()
        pickled_item['label'] = label
        pickled_item['fixed_len_array'] = fixed_len_array
        pickled_item['var_len_array'] = var_len_array

        data_dict[str(i)] = pickled_item
        i_local += 1

    fp_syr.close()

    return data_dict


data_dict = create_test_data(BASE_PATH, NUM_ITEMS, MAX_ITEMS)


class TestSingleDataset(unittest.TestCase):
    def test_sequential_item_read(self):
        syr_dataset = SyrahDataset(BASE_PATH + '_0' + '.syr', keys=['idx', 'label', 'fixed_len_array', 'var_len_array'])
        syr_dataloader = DataLoader(syr_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                    worker_init_fn=syr_dataset.open)

        for idx, label, fixed_len_array, var_len_array in syr_dataloader:
            pickled_item = data_dict[str(idx[0].numpy()[0])]
            assert np.all(label.numpy() == pickled_item['label'])
            assert np.all(fixed_len_array.numpy() == pickled_item['fixed_len_array'])
            assert np.all(var_len_array.numpy() == pickled_item['var_len_array'])


class TestConcatDataset(unittest.TestCase):
    def test_sequential_item_read(self):
        syr_dataset_0 = SyrahDataset(BASE_PATH + '_0' + '.syr', keys=['idx', 'label', 'fixed_len_array', 'var_len_array'])
        syr_dataset_1 = SyrahDataset(BASE_PATH + '_1' + '.syr', keys=['idx', 'label', 'fixed_len_array', 'var_len_array'])
        syr_dataset = SyrahConcatDataset([syr_dataset_0, syr_dataset_1])
        syr_dataloader = DataLoader(syr_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                    worker_init_fn=syr_dataset.open)

        for idx, label, fixed_len_array, var_len_array in syr_dataloader:
            pickled_item = data_dict[str(idx[0].numpy()[0])]
            assert np.all(label.numpy() == pickled_item['label'])
            assert np.all(fixed_len_array.numpy() == pickled_item['fixed_len_array'])
            assert np.all(var_len_array.numpy() == pickled_item['var_len_array'])


if __name__ == '__main__':
    unittest.main()
