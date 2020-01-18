# reference https://docs.python.org/3.7/extending/building.html#building
from distutils.core import setup, Extension

module = Extension('keywdarg', sources = ['keywdarg.c'])

setup (name = 'Keydarg',
       version = '1.0',
       description = 'This is a demo package',
       ext_modules = [module])

# running
# python setup.py build
