"""
Class for estimation of information dynamics of time-dependent probabilistic document representations.

This method will be deprecated. New implementation can be found in
https://github.com/centre-for-humanities-computing/infodynamics/
"""
import json
import numpy as np
from .metrics import kld, jsd


class InfoDynamics:
    def __init__(self, data, time, window=3, weight=0, sort=False, normalize=False):
        """
        - data: list/array (of lists), bow representation of documents
        - time: list/array, time coordinate for each document (identical order as data)
        - window: int, window to compute novelty, transience, and resonance over
        - weight: int, parameter to set initial window for novelty and final window for transience
        - sort: bool, if time should be sorted in ascending order and data accordingly
        - normalize: bool, make row sum to 1
        """
        self.window = window
        self.weight = weight

        if sort:
            self.data = np.array([text for _, text in sorted(zip(time, data))])
            self.time = sorted(time)
        else:
            self.data = np.array(data)
            self.time = time

        self.m = self.data.shape[0]

        if normalize:
            data = data / data.sum(axis=1, keepdims=True)

    def novelty(self, meas=kld):
        N_hat = np.zeros(self.m)
        N_sd = np.zeros(self.m)
        for i, x in enumerate(self.data):
            submat = self.data[(i - self.window):i, ]
            tmp = np.zeros(submat.shape[0])
            if submat.any():
                for ii, xx in enumerate(submat):
                    tmp[ii] = meas(x, xx)
            else:
                tmp = np.zeros([self.window]) + self.weight

            N_hat[i] = np.mean(tmp)
            N_sd[i] = np.std(tmp)

        self.nsignal = N_hat
        self.nsigma = N_sd

    def transience(self, meas=kld):
        T_hat = np.zeros(self.m)
        T_sd = np.zeros(self.m)
        for i, x in enumerate(self.data):
            submat = self.data[i + 1:(i + self.window + 1), ]
            tmp = np.zeros(submat.shape[0])
            if submat.any():
                for ii, xx in enumerate(submat):
                    tmp[ii] = meas(x, xx)
            else:
                tmp = np.zeros([self.window])

            T_hat[i] = np.mean(tmp)
            T_hat[-self.window:] = np.zeros([self.window]) + self.weight
            T_sd[i] = np.std(tmp)

        self.tsignal = T_hat
        self.tsigma = T_sd

    def resonance(self, meas=kld):
        self.novelty(meas)
        self.transience(meas)
        self.rsignal = self.nsignal - self.tsignal
        self.rsignal[:self.window] = np.zeros([self.window]) + self.weight
        self.rsignal[-self.window:] = np.zeros([self.window]) + self.weight
        self.rsigma = (self.nsigma + self.tsigma) / 2
        self.rsigma[:self.window] = np.zeros([self.window]) + self.weight
        self.rsigma[-self.window:] = np.zeros([self.window]) + self.weight

    def slice_zeros(self):
        self.nsignal = self.nsignal[self.window:-self.window]
        self.nsigma = self.nsigma[self.window:-self.window]
        self.tsignal = self.tsignal[self.window:-self.window]
        self.tsigma = self.tsigma[self.window:-self.window]
        self.rsignal = self.rsignal[self.window:-self.window]
        self.rsigma = self.rsigma[self.window:-self.window]

    def fit(self, meas, slice_w=False):
        self.novelty(meas)
        self.transience(meas)
        self.resonance(meas)
        if slice_w:
            self.slice_zeros()

    def fit_save(self, meas, path, slice_w=False):
        self.fit(meas, slice_w)

        out = {
            'novelty': self.nsignal.tolist(),
            'novelty_sigma': self.nsigma.tolist(),
            'transience': self.tsignal.tolist(),
            'transience_sigma': self.tsigma.tolist(),
            'resonance': self.rsignal.tolist(),
            'resonance_sigma': self.rsigma.tolist(),
        }

        with open(path, 'w') as f:
            json.dump(out, f)
