"""
    Configuration file

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
