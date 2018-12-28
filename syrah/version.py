from collections import namedtuple

_SYRAH_VERSION = namedtuple("_SYRAH_VERSION", "major minor bugfix")

version_tuple = _SYRAH_VERSION(0, 1, 1)

version = "{0.major:d}.{0.minor:d}.{0.bugfix:d}".format(version_tuple)
