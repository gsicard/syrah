"""
    Configuration file
"""

"""
    Magic byte for format identification
"""
MAGIC_BYTES = b'GS'

"""
    Number of bytes in which to store the version number.
"""
NUM_BYTES_VERSION = 4
"""
    Number of bytes in which to store the position of the first byte of the serialized metadata.
"""
NUM_BYTES_METADATA_OFFSET = 8
"""
    Number of bytes in which to store the length of the serialized metadata.
"""
NUM_BYTES_METADATA_LENGTH = 8
"""
    Number of magic bytes.
"""
NUM_BYTES_MAGIC_BYTES = len(MAGIC_BYTES)
