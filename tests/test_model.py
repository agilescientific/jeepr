# -*- coding: utf 8 -*-
"""
Define a suite a tests for jeepr.
"""
import numpy as np

from jeepr import Model
from jeepr import utils


def test_vti():
    model = Model.from_gprMax_vti('tests/Cylinder_halfspace.vti')
    assert model.shape == (105, 120)
    assert np.allclose(model.mean(), 1.79682540)


def test_conversion():
    model = Model.from_gprMax_vti('tests/Cylinder_halfspace.vti')
    v = np.ones(model.shape) * 5
    vsoil = utils.c / np.sqrt(v)
    dt = 5e-11  # 50 ps
    m_time, t = model.to_time(vsoil, dt=dt)
    assert np.allclose(m_time.dt, dt)
    assert np.allclose(t[-1], 3.1e-9)
    assert np.allclose(t[-1]*1e9, m_time.extent[2])
