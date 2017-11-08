# -*- coding: utf 8 -*-
"""
==================
jeepr
==================
"""
from .scan import Scan
from .model import Model

__all__ = [
           'Scan',
           'Model',
          ]

__version__ = "unknown"
try:
    from ._version import __version__
except ImportError:
    pass
