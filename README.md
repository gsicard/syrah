**Important note:** this code is still in development, and while all documented features have been properly tested, it should still be used with caution.
# Syrah
Syrah (simple random access dataset format) allows for fast random access of on-disk arbitrary-type indexed arrays.

It was designed as a backend for PyTorch dataset API since other options were deemed too slow or required to load the entire file in memory.

Syrah works by loading the dataset metadata in memory (arrays type, position in file and length) and using it to rapidly access and deserialize arrays from the file on disk. 

This package contains the high level implementation to read and write syrah files as well as wrappers for the PyTorch dataset API.

<!-- TOC -->
- [1. Installation](#1-installation)
- [2. Writing to a syrah file](#2-writing-to-a-syrah-file)
- [3. Reading from a syrah file](#3-reading-from-a-syrah-file)
- [4. PyTorch dataset API](#4-pytorch-dataset-api)
- [5. Multiprocessing concurrency issues](#5-multiprocessing-concurrency-issues)
- [6. File format summary](#6-file-format-summary)
<!-- /TOC -->

## 1. Installation


```bash
git clone https://github.com/gsicard/syrah.git
cd syrah
pip install --upgrade .
```

## 2. Writing to a syrah file

Let us first create a random dataset of fixed size float feature vectors and their corresponding binary labels:

```python
import numpy as np
from syrah import File

num_samples = 10_000
num_features = 1_000

features = np.random.random(size=(num_samples, num_features))
labels = np.random.randint(2, size=(num_samples, 1))
```

Each sample is added as a dictionary of arrays using the `File.add_item()` method:

```python
file_path = '/tmp/test.syr'

with File(file_path, mode='w') as syr:
    for i in range(num_samples):
        syr.add_item({'label': labels[i], 'features': features[i]})
```

**Important note:** if a syrah `File` is opened in writing mode outside a context manager (`with` statement) it needs to be closed explicitly using the `File.close()` method to ensure that the metadata is written at the end of the file and that the headers are properly updated.
 
## 3. Reading from a syrah file

Similarly to writing, the whole sample can be read at once and returns a dictionary of arrays:

```python
with File(file_path, mode='r') as syr:
    for i in range(syr.num_items()):
        # item is a dictionary with keys 'label' and 'features'
        item = syr.get_item(i)
```

Or each array can be read independently:

```python
with File(file_path, mode='r') as syr:
    for i in range(num_samples):
        label = syr.get_array(i, 'label')
        features = syr.get_array(i, 'features')
```

## 4. PyTorch dataset API

If the data does not need to be preprocessed before being fed to the network, a `SyrahDataset` object can be used with a PyTorch `Dataloader`:

```python
from syrah.utils.data import SyrahDataset
from torch.utils.data import DataLoader

dataset = SyrahDataset(file_path, ['features', 'label'])
data_generator = DataLoader(dataset, batch_size=32, shuffle=True)

for features, labels in data_generator:
    ...
```

If data preprocessing is needed, the `__init__()` and `__getitem__()` methods should be overridden:
```python
class CustomSyrahDataset(SyrahDataset):
    def __init__(self, file_path: str, *args):
        """
        Create a new `Dataset` object.
        :param file_path: path to the Syrah file
        :param args: list of extra arguments required for the preprocessing
        """
        super().__init__(file_path, ['features', 'label'])
        ...

    def __getitem__(self, item: int) -> Tuple[ndarray, ndarray]:
        """
        Access the item at the  specified index
        :param item: index of the item
        :return: features and corresponding label
        """
        features, label = super().__getitem__(item)
        features = ...
        
        return features.astype(np.float32), label.astype(np.float32)
```

If the dataset consists of multiple syrah files, a `SyrahConcatDataset` object should be created from a list of `SyrahDataset` objects:

```python
from syrah.utils.data import SyrahConcatDataset

concat_dataset = SyrahConcatDataset([dataset_1, ...])
```

## 5. Multiprocessing concurrency issues

Syrah high level API supports multiprocess read access but each worker needs to call the `File.open()` method to get his own file object and avoid concurrency issues when reading from multiple processes:

```python
from multiprocessing import Pool

num_workers = 8
syr = File(file_path, mode='r')


def read_item(i):
    item = syr.get_item(str(i))


p = Pool(num_workers, initializer=syr.open, initargs=(file_path, 'r'))
p.map(read_item, range(num_samples))
```

Similarly, when using `num_workers > 0` with PyTorch `Dataloader`, `SyrahDataset.open()` also needs to be called by each worker (or `SyrahConcatDataset.open()` in case of multiple syrah files):

```python
data_generator_multi = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=num_workers, worker_init_fn=dataset.open)

for features, labels in data_generator_multi:
    ...
```

## 6. File format summary

The file format is as follows:
- headers (first 22 bytes):
    - 2 bytes for the format "magic bytes" ("\x47\x53")
    - 4 bytes for storing the version number (4 uint8: 0 followed by x, y, z as in version x.y.z)
    - 8 bytes for the serialized metadata offset (int64)
    - 8 bytes for the serialized metadata length (int64)
- data (arbitrary size):
    - concatenation of byte representations of all arrays in the dataset
- metadata (arbitrary size):
    - serialized metadata using bson
