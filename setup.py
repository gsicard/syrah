from setuptools import setup

torch = ['torch>=1.0.0']
all = torch

extras_require = {
    'all': all,
    'torch': torch
}

setup(
    name='pysyrah',
    version='0.2.0-dev',
    packages=['syrah', 'syrah._hl', 'syrah.utils'],
    url='https://github.com/gsicard/syrah',
    license='GNU General Public License v3 (GPLv3)',
    author='gsicard',
    description='Simple random access dataset format',
    install_requires=[
        'bson>=0.5.7',
        'numpy>=1.12.0'
    ],
    extras_require=extras_require
)
