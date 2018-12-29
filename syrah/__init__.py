"""
    Python implementation of Simple Random Access Dataset Format.

    This format is designed to store indexed dictionaries of Numpy arrays (called items) on disk
    while allowing for fast random access of the items from their index using an in-memory index table.
    It supports both fixed length and variable length array storage.

    The data is stored in a binary file as follows:
    <FORMAT VERSION><METADATA OFFSET><METADATA LENGTH><MAGIC BYTES><BYTE ARRAYS><SERIALIZED METADATA>
"""

# TODO: add import error checking
# TODO: add setup.py
from ._hl.files import File
from .version import version as __version__
