"""
    Syrah versioning

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
from collections import namedtuple

_SYRAH_VERSION = namedtuple("_SYRAH_VERSION", "major minor bugfix")

version_tuple = _SYRAH_VERSION(0, 3, 0)

version = "{0.major:d}.{0.minor:d}.{0.bugfix:d}".format(version_tuple)
