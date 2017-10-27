# -*- coding: utf 8 -*-
"""
Define a suite a tests for jeep.
"""
from jeep import Scan

FNAME = 'tests/Multi-01_LINE00A.RA1'


def test_load():

    scan = Scan.from_rad(FNAME)
    assert scan.shape == (233, 256)
