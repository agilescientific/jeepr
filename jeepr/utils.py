# -*- coding: utf-8 -*-
'''
Utilities.
'''
import numpy as np


c = 299792458  # speed of light in vacuum.


def srange(start, step, length, dtype=None):
    """
    Like np.arange() but you give the start, the step, and the number
    of steps. Saves having to compute the end point yourself.
    """
    stop = start + (step * length)
    return np.arange(start, stop, step, dtype)


def find_nearest(arr, value, return_index=False):
    idx = (np.abs(arr - value)).argmin()
    if return_index:
        return idx
    return arr[idx]


def get_lines(fname, param):
    """
    Get particular lines from the .in file.
    """
    with open(fname) as f:
        lines = f.readlines()
    out = []
    for line in lines:
        if line.startswith('#{}:'.format(param)):
            out.append(line.split(':')[1])
    return out


def get_trace_spacing(fname):
    """
    Returns the trace spacing between runs in simulation (m).
    Fname is an .in file.
    """
    info, = get_lines(fname, 'rx_steps')
    return float(info.split()[0])


def get_gprMax_materials(fname):
    """
    Returns the soil permittivities. Fname is an .in file.
    """
    materials = {'pec': 1.0,  # Not defined, usually taken as 1.
                 'free_space': 1.000536}

    for mat in get_lines(fname, 'material'):
        props = mat.split()
        materials[props[-1]] = float(props[0])

    return materials


def get_freq(fname):
    """
    Returns freq from infile.
    """
    info, = get_lines(fname, 'waveform')
    return info.split()[2]


def vel_from_perm(model, nano=False):
    """
    Converts an array of relative permitivities into
    velocity (metres per second)
    """
    if nano:
        return 1e-9 * c / np.sqrt(model)
    else:
        return c / np.sqrt(model)
