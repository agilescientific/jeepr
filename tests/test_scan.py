# -*- coding: utf 8 -*-
"""
Define a suite a tests for jeepr.
"""
import numpy as np
import pytest

from jeepr import Scan
from jeepr.scan import ScanError

scan = Scan.from_gprmax('tests/test_2D_merged.out')


def test_rad():
    scan = Scan.from_rad('tests/Multi-01_LINE00A.RA1')
    assert scan.shape == (256, 233)


def test_gprmax():
    assert scan.shape == (1358, 85)
    assert np.allclose(scan.time[-1], [1.6003469675e-08])


def test_condition():
    d_ = scan.demean()
    assert d_.log[-1].startswith('demean')

    g_ = d_.gain()
    assert g_.log[-1].startswith('linear')
    assert np.allclose(g_.mean(), 3.976527829e-17)
    assert np.allclose(d_.gain('power').mean(), 1.1818782402e-17)


def test_spectrum():
    ps = [-26.73432684, -22.27886143, -26.81604849,  -8.06285634,  -3.6589573]
    f, p = scan.T[0].get_spectrum()
    assert np.allclose(p[:5], ps)


def test_crop_idx():
    s_ = scan.crop(idx=10)
    assert s_.log[-1].startswith('crop')


def test_crop_t():
    s_ = scan.crop(t=10, reset_t0=False)
    assert s_.log[-1].startswith('crop')
    assert np.allclose(s_.time[0], 10e-9)


def test_resample():
    dt_new = 2e-10
    s_ = scan.resample(dt=dt_new)
    assert np.allclose(s_.time[1], dt_new)


def test_crop_error():
    with pytest.raises(ScanError):
        scan.crop()
