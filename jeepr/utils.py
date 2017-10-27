# -*- coding: utf-8 -*-
'''
Utilities.
'''
import re
import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup
import h5py
import vtk
from vtk.util.numpy_support import vtk_to_numpy

matplotlib.use('agg')

c = 299792458  # speed of light in vacuum.


def _get_array(output, idx):
    """
    Get an array from VTK output data, then reshape and
    flip it to use in NumPy and plot in matplotlib.
    """
    flat = vtk_to_numpy(output.GetCellData().GetArray(idx))
    x, y, z = np.array(output.GetDimensions()) - 1
    return np.flipud(flat.reshape((y, x, z)))


def _get_meta(fname, shape, spacing):
    """
    Get some metadata from the XML of a VTK file, and
    combine it with some key data about the grid.

    Probably makes a bunch of assumptions about the
    structure of the XML file.
    """
    with open(fname, 'rb') as f:
        data = str(f.read())

    meta = re.search(r'<gprMax>(.+?)</gprMax>', data).group(0)
    soup = BeautifulSoup(meta, "xml")

    legend = {"Materials": [{'name': m['name'], 'value':int(m.contents[0])}
                            for m
                            in soup.find_all("Material")]}

    for e in ["PML", "Sources", "Receivers"]:
        legend[e] = {}
        legend[e]['name'] = soup.find(e)['name']
        legend[e]['value'] = int(soup.find(e).contents[0])

    pattern = re.compile(r'\(([0-9]+),([0-9]+),([0-9]+)\)')
    tx = legend['Sources']['name']
    spos = spacing * np.array([int(s)
                              for s
                              in pattern.search(tx).groups()])
    legend['Sources']['position'] = spos
    legend['Sources']['type'] = tx[:tx.find('(')]

    rx = legend['Receivers']['name']
    rpos = spacing * np.array([int(s)
                              for s
                              in pattern.search(rx).groups()])
    legend['Receivers']['position'] = rpos

    y, x, z = shape
    legend['Model'] = {'Shape': (x, y, z),
                       'Domain': spacing * np.array([x, y, z]),
                       'Spacing': spacing}

    return legend


def get_data(fname):
    """
    Get data and metadata from a VTK file.

    Assumes gprMax 'material' array has index 0, and
    gprMax PML/Sources array has index 1.
    """
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(fname)
    reader.Update()

    output = reader.GetOutput()
    spacing = reader.GetOutput().GetSpacing()

    material = _get_array(output, 0)
    pml = (_get_array(output, 1) == 1).astype(int)
    legend = _get_meta(fname, pml.shape, spacing)

    return material, pml, legend


def get_lines(fname, param):
    """
    Get particular lines from the .in file.
    """
    with open(fname) as f:
        lines = f.readlines()
    out = []
    for line in lines:
        if line.startswith('#'+param):
            out.append(line.split(':')[1])
    return out


def get_trace_spacing(fname):
    """
    Returns the trace spacing between runs in simulation (m).
    Fname is an .in file.
    """
    info, = get_lines(fname, 'rx_steps')
    return float(info.split()[0])


def get_materials(fname):
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


def intvel_to_avgvel(data):
    """
    Takes and 2d array of interval velocities and computes a vavg velocity
    model (average down to each cell) for depthing conversion.
    """
    ny, nx = data.shape
    vacc = np.add.accumulate(data, axis=0)
    x = np.ones(nx)
    y = np.linspace(1, ny+1, ny)
    xv, yv = np.meshgrid(x, y)
    return vacc/yv


def plot_section(section, size=(10, 10), color='viridis'):
    """
    Helper function to plot a 2D array size: is figsize in inches
    returns a new figure.
    """
    fig = plt.figure(figsize=size)
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.8])
    im = ax.imshow(section, cmap=color)
    fig.colorbar(im, shrink=0.35)  # ticks=[-1, 0, 1])


def get_bscan_section(filename):
    """
    Returns a tuple where the first element is the 2D numpy array
    corresponding to the Bscan profile, and the second element is the time
    basis for the vertical axis.
    """
    f = h5py.File(filename, 'r')
    nrx = f.attrs['nrx']
    dt = f.attrs['dt']
    iterations = f.attrs['Iterations']

    # Get scanned data from file
    for rx in range(1, nrx + 1):
        path = '/rxs/rx' + str(rx) + '/' + 'Ez'
        outputdata = f[path][:]

    # Make a time-axis
    t = np.linspace(0, 1, iterations)
    t *= (iterations * dt)
    return outputdata, t


def plot_earth_model(geom_name, save_fig=True):
    """
    Plot the model.
    """
    mat, pml, legend = get_data(geom_name)

    xmax = legend['Model']['Domain'][0]
    ymax = legend['Model']['Domain'][1]

    mat_ = mat[..., 0]
    matf = mat_.astype('float32')
    src_pos = legend['Sources']['position']
    rx_pos = legend['Receivers']['position']

    uniq = np.unique(mat_)
    cmap = plt.get_cmap('viridis', uniq.size)
    mi, ma = np.min(mat_), np.max(mat_)

    # Display.
    plt.figure(figsize=(15, 12))
    mat_plot = plt.imshow(matf,
                          cmap=cmap,
                          vmin=mi-.5,
                          vmax=ma+.5,
                          extent=[0, xmax, 0, ymax])

    # Colourbar.
    cbar = plt.colorbar(mat_plot, shrink=0.3)
    cbar.set_ticks(uniq)

    # Plot src and rcv.
    plt.plot(*src_pos[:2], 'or')
    plt.plot(*rx_pos[:2], 'ob')

    # Axes.
    plt.title("Earth model")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.grid()

    if save_fig:
        plt.savefig(geom_name[:-4]+'.png')

    return
