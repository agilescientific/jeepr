# -*- coding: utf-8 -*-
"""
Defines scan objects.

:copyright: 2017 Agile Geoscience
:license: Apache 2.0
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from . import utils
from .rad2np import read_rad


class ScanError(Exception):
    """
    Generic error class.
    """
    pass


class Scan(np.ndarray):
    """
    A fancy ndarray. Gives some utility functions, plotting, etc, for scans.
    """
    def __new__(cls, data, params):
        """
        I am just following the numpy guide for subclassing ndarray...
        """
        obj = np.asarray(data).view(cls).copy()

        params = params or {}

        for k, v in params.items():
            setattr(obj, k, v)

        return obj

    def __array_finalize__(self, obj):
        """
        I am just following the numpy guide for subclassing ndarray...
        """
        if obj is None:
            return

        if obj.size == 1:
            return float(obj)

        self.name = getattr(obj, 'name', '')

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    @classmethod
    def from_rad(cls, fname):
        """
        Constructor for USRadar RAD files, extension RA1, RA2 or RAD.

        Args:
            fname (str): a RAD file.

        Returns:
            scan. The scan object.
        """
        params, _, arr = read_rad(fname)

        return cls(arr, params)
