# -*- coding: utf 8 -*-
"""
Define a suite a tests for jeepr.
"""
import numpy as np

from jeepr import Model


def test_vti():
    model = Model.from_vti('tests/Cylinder_halfspace.vti')
    assert model.shape == (105, 120)
    assert np.allclose(model.mean(), 1.79682540)
