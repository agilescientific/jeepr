# -*- coding: utf 8 -*-
"""
Define a suite a tests for jeepr.
"""
import numpy as np

from jeepr import Scan


def test_rad():
    scan = Scan.from_rad('tests/Multi-01_LINE00A.RA1')
    assert scan.shape == (256, 233)


def test_gprmax():
    scan = Scan.from_gprmax('tests/gprMax_Bscan_merged.out')
    assert scan.shape == (637, 60)
    assert np.allclose(scan.time[-1], [3.00020832e-9])


def test_spectrum():
    scan = Scan.from_gprmax('tests/gprMax_Bscan_merged.out')
    ps = [-37.36291442, -17.34384542,  -8.04115503,  -2.41136352,  -1.05919635]
    f, p = scan.T[0].get_spectrum()
    assert np.allclose(p[:5], ps)
