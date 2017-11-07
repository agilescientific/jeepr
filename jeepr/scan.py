# -*- coding: utf-8 -*-
"""
Defines scan objects.

:copyright: 2017 Agile Geoscience
:license: Apache 2.0
"""
import random
import copy

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import h5py
from bruges.transform import time_to_depth

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

        # The atrtibutes you want child objects to inherit, eg after slicing.
        # This could go wrong... may need some sanity checking.
        self.freq = getattr(obj, 'freq', 0)
        self.t0 = getattr(obj, 't0', 0)
        self.x0 = getattr(obj, 'x0', 0)
        self.dt = getattr(obj, 'dt', 0)
        self.dx = getattr(obj, 'dx', 0)
        self.dz = getattr(obj, 'dz', 0)
        self.log = getattr(obj, 'log', [])
        self.domain = getattr(obj, 'domain', 0)
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
        meta, _, arr = read_rad(fname)
        dt = meta['SPR_SAMPLING_INTERVAL']  # in picoseconds
        dx = meta['SPR_SHAFT_INTERVAL']     # in metres...

        params = {'dt': float(dt) * 1e-12,
                  'dx': float(dx),
                  'x0': 0.0,
                  'domain': 'time',
                  'meta': meta,
                  'log': ['loaded from rad'],
                  }

        return cls(arr.astype(np.float).T, params)

    @classmethod
    def from_gprmax(cls, fname, infile=None):
        """
        Constructor for merged gprMax .out files.

        Args:
            fname (str): a gprMax OUT file.
            infile (str): the associated .IN file, or leave blank to find
                most likely name.

        Returns:
            scan. The scan object.
        """
        f = h5py.File(fname, 'r')
        nrx = f.attrs['nrx']
        dt = f.attrs['dt']
        # iterations = f.attrs['Iterations']

        # Get scanned data from file
        for rx in range(1, nrx + 1):
            path = '/rxs/rx' + str(rx) + '/' + 'Ez'
            arr = f[path][:]

        params = {'dt': dt,
                  't0': 0.0,
                  'domain': 'time',
                  'meta': {'nrx': nrx},
                  'log': ['loaded from out'],
                  }

        # Get wavelet and other params.
        if infile is None:
            infile = fname.replace('_merged.out', '.in')
        x, = utils.get_lines(infile, 'waveform')
        params['freq'] = float(x.split()[-2])
        x, = utils.get_lines(infile, 'rx_steps')
        params['dx'] = float(x.split()[0])
        x, = utils.get_lines(infile, 'rx')
        params['x0'] = float(x.split()[0])

        return cls(arr.astype(np.float), params)

    @property
    def time(self):
        """
        The time basis of the scan. Computed from dt and shape.
        """
        return utils.srange(self.t0, self.dt, self.shape[0])

    @property
    def nx(self):
        """
        The number of x samples.
        """
        try:
            return self.shape[1]
        except IndexError:
            return 1

    @property
    def x(self):
        """
        The spatial basis of the scan.
        """
        return utils.srange(self.x0, self.dx, self.nx)

    @property
    def extent(self):
        """
        The extent. Useful for ``plt.imshow(scan, extent=extent)``.
        """
        return [self.x[0], self.x[-1], self.time[-1]*1e9, self.time[0]*1e9]

    def crop(self, t=None, idx=None, reset_t0=True):
        """
        Crop a scan at some time.

        For now nearest, should resample.
        """
        params = copy.deepcopy(self.__dict__)
        if (idx is None) and (t is None):
            raise ScanError("You must provide t or idx for the crop.")
        if (t is not None) and t > 1:
            t *= 1e-9  # Convert nanoseconds to seconds.
        if idx is None:
            idx = utils.find_nearest(self.time, t, return_index=True)
        if t is None:
            t = self.time[idx]
        if reset_t0:
            params['t0'] = 0.0
        else:
            params['t0'] = t
        params['log'].append('crop at t = {:.2f} ns'.format(t*1e9))
        return Scan(self[idx:, ...], params)

    def plot(self, ax=None, return_fig=False, percentile=99, alpha=1.0):
        """
        Plot a scan.

        Args:
            ax (ax): A matplotlib axis.
            return_fig (bool): whether to return the matplotlib figure.
                Default False.

        Returns:
            ax. If you passed in an ax, otherwise None.
        """
        if ax is None:
            fig = plt.figure(figsize=(8, 4.5))
            ax = fig.add_subplot(111)
            return_ax = False
        else:
            return_ax = True

        ma = np.percentile(self, percentile)
        im = ax.imshow(self,
                       origin='upper',
                       extent=self.extent,
                       aspect='auto',
                       vmin=-ma, vmax=ma,
                       alpha=alpha,
                       cmap="Greys",
                       )
        _ = plt.colorbar(im)
        ax.set_xlabel('x position (m)', size=16)
        if self.domain == 'time':
            ax.set_ylabel('two-way time (ns)', size=16)
        else:
            ax.set_ylabel('estimated depth (m)', size=16)
        ax.set_title(self.name, size=16)

        if return_ax:
            return ax
        elif return_fig:
            return fig
        else:
            return None

    def resample(self, dt, window=None):
        """
        Resamples a scan in time.

        Args:
            dt (float): The new sample interval in seconds.
            window (array_like, callable, string, float, or tuple). Use
                according to ageo.co/2h0SCn5 and ageo.co/2iXHuvr.
        """
        params = copy.deepcopy(self.__dict__)
        dt_old = params['dt']
        window = window or "hamming"
        num = int(1 + self.time[-1] / dt)
        if num < 2:
            raise ScanError("Resampling would result in too few samples.")
        arr = scipy.signal.resample(self, num, axis=0, window=window)
        params['dt'] = dt
        params['log'].append('resampled {} s to {} s'.format(dt_old, dt))
        return Scan(arr, params)

    def add_model(self, arr, params):
        """
        Add a model from an array.

        Args:
            arr (ndarray): An array of ints.
            params (dict): A dict mapping ints to dicts of properties.
            fname (str): a RAD file.

        Returns:
            None.
        """
        pass

    def demean(self):
        """
        Subtracts the mean traces from the scan.

        Returns:
            Scan.
        """
        params = copy.deepcopy(self.__dict__)
        arr = self - np.mean(self, axis=1, keepdims=True)
        params['log'].append('demeaned')
        return Scan(arr, params)

    def gain(self, method=None):
        """
        Applies one of a few gain functions.

        Args:
            method (str): Which method to use. "linear", "reciprocal",
                or "power".

        Returns:
            Scan.
        """
        params = copy.deepcopy(self.__dict__)
        nt = self.shape[0]
        method = method or "linear"
        if method == "linear":
            f = (np.arange(nt)/nt).reshape(nt, 1)
        elif method == "reciprocal":
            f = ((1 - 1/np.arange(nt))**12).reshape(nt, 1)
        elif method == "power":
            f = ((np.arange(nt)/512)**2).reshape(nt, 1)
        else:
            raise ScanError("Method must be linear, reciprocal or power.")
        arr = self * f
        params['log'].append('{} gain'.format(method))
        return Scan(arr, params)

    def to_depth(self, v, dz=0.001):
        """
        Convert Scan to depth domain.

        Args:
            v (ndarray): Interval velocities. Must be same shape as model. If
                None, then the model's own permittivities are used.
            dz (float): The new depth step or sampling interval, in m.

        Returns:
            Model. The depth-converted model, as a jeepr.Model instance.
        """
        if self.domain != 'time':
            raise ScanError('You can only depth-convert a time scan.')
        params = copy.deepcopy(self.__dict__)
        arr = time_to_depth(self, v, dz=dz, dt=self.dt)
        basis = utils.srange(0, dz, arr.shape[0])
        params['domain'] = 'depth'
        params['dz'] = dz
        params['dt'] = 0
        return Scan(arr, params), basis

    def get_spectrum(self, n=None, return_amp=False):
        """
        Get a power spectrum for the scan.

        Args:
            n (int): Number of traces to use, default 10% or at least 10.
            return_amp (bool): Whether to return amplitude instead of power.

        Returns:
            tuple: (ndarray, ndarray), the frequencies and power (or amplitude)
                values. Often, you want to do plt.plot(f, p) next.
        """
        if n is None:
            n = 0.1
        if n < 1:
            n = max(int(n*self.nx), 10)
        if self.nx < 10:
            n = self.nx
        traces = random.sample(range(self.nx), n)

        amp = []

        if n > 1:
            trace_list = self.T[traces]
        else:
            trace_list = [self]
        for i, tr in enumerate(trace_list):
            tr = tr.astype(np.float) * np.blackman(len(tr))
            amp.append(np.abs(np.fft.rfft(tr)))

        f = np.fft.rfftfreq(len(tr), d=self.dt)
        a = np.mean(amp, axis=0)
        p = 20 * np.log10(a)
        p -= np.amax(p)  # Normalize to 0.

        if return_amp:
            return f, a
        else:
            return f, p
