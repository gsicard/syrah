from setuptools import setup

setup(
    name='syrah',
    version='0.1.2',
    packages=['syrah', 'syrah._hl', 'syrah.utils'],
    url='https://github.com/gsicard/syrah',
    license='',
    author='gsicard',
    description='Simple random access dataset format',
    install_requires=[
        'bson>=0.5.7',
        'pytorch>=1.0.0',
        'numpy>=1.12.0'
    ]
)
