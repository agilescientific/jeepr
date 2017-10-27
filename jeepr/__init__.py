# -*- coding: utf 8 -*-
"""
==================
jeepr
==================
"""
from .scan import Scan

__all__ = [
           'Scan',
          ]

__version__ = "unknown"
try:
    from ._version import __version__
except ImportError:
    pass
