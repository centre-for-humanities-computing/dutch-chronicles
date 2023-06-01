"""
Class for estimation of information dynamics of time-dependent probabilistic document representations
    taken from https://github.com/centre-for-humanities-computing/newsFluxus/blob/master/src/tekisuto/models/infodynamics.py
    commit 1fb16bc91b99716f52b16100cede99177ac75f55
"""
import json
import numpy as np
from ..metrics import kld, jsd
from tqdm import tqdm

class InfoDynamics:
    def __init__(self, data, time, window=3, weight=0, sort=False, normalize=False, group_windows=False):
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
            self.data = np.array([text for _,text in sorted(zip(time, data))])
            self.time = sorted(time)
        else:
            self.data = np.array(data)
            self.time = time

        self.m = self.data.shape[0]
        self.group_windows = group_windows
    
        if normalize:
            data = data / data.sum(axis=1, keepdims=True)
    
    def _get_slices_novelty(self,i,window,timestamps,time2index,index2time):
        target_date = timestamps[i]
        window_delim_dateindex = time2index[target_date] - window
        window_delim_dateindex = 0 if window_delim_dateindex < 0 else window_delim_dateindex
        window_delim_date = index2time[window_delim_dateindex]
        window_delim_index = timestamps.index(window_delim_date)
        return window_delim_index

    def _get_slices_transience(self,i,window,timestamps,time2index,index2time):
        target_date = timestamps[i]
        window_delim_dateindex = time2index[target_date] + window + 1
        window_delim_dateindex = time2index[timestamps[-1]] if window_delim_dateindex > time2index[timestamps[-1]] else window_delim_dateindex
        window_delim_date = index2time[window_delim_dateindex]
        window_delim_index = timestamps.index(window_delim_date) - 1
        return window_delim_index


    def novelty(self, meas=kld):
        N_hat = np.zeros(self.m)
        N_sd = np.zeros(self.m)
        pointwise_r = np.zeros(self.data.shape)

        if self.group_windows == True:
            time2index = {d:c for c,d in enumerate(sorted(list(set(self.time))))}
            index2time = {c:d for d,c in time2index.items()}
            slices = [self._get_slices_novelty(i=i,window=self.window,timestamps=self.time,time2index=time2index,index2time=index2time) for i in range(self.data.shape[0])]
            self.nslices = slices 


        if self.group_windows == False:
            slices = [i - self.window for i in np.arange(self.m)]
            slices = [0 if w < 0 else w for w in slices]
            self.nslices = slices 
            
        for i, x in tqdm(enumerate(self.data)):
            w = slices[i]
            submat = self.data[w:i,]
            tmp = np.zeros(submat.shape[0])
            tmp_pw = np.zeros(submat.shape)

            if submat.any():
                for ii, xx in enumerate(submat):
                    tmp[ii] = meas(x, xx)
                    tmp_pw[ii,:] = np.where(x != 0, (x) * np.log10(x / xx), 0)
                pointwise_r[i,:] = np.mean(tmp_pw,axis=0)
            else:
                tmp = np.zeros([self.window]) + self.weight
                pointwise_r[i,:] = np.arange(self.data.shape[1])
            
            N_hat[i] = np.mean(tmp)
            N_sd[i] = np.std(tmp)
                
        self.nsignal = N_hat
        self.nsigma = N_sd
        self.npointwise = pointwise_r

    def transience(self, meas=kld):
        T_hat = np.zeros(self.m)
        T_sd = np.zeros(self.m)
        pointwise_r = np.zeros(self.data.shape)

        if self.group_windows == True:
            time2index = {d:c for c,d in enumerate(sorted(list(set(self.time))))}
            index2time = {c:d for d,c in time2index.items()}
            slices = [self._get_slices_transience(i=i,window=self.window,timestamps=self.time,time2index=time2index,index2time=index2time) for i in range(self.data.shape[0])]
            self.tslices = slices

        if self.group_windows == False:
            slices = [i + self.window for i in np.arange(self.m)]
            slices = [self.m if w > self.m else w for w in slices]
            self.tslices = slices
        
        for i, x in tqdm(enumerate(self.data)):
            w = slices[i]
            submat = self.data[i:w,]
            tmp = np.zeros(submat.shape[0])
            tmp_pw = np.zeros(submat.shape)

            if submat.any():
                for ii, xx in enumerate(submat):
                    tmp[ii] = meas(x, xx)
                    tmp_pw[ii,:] = np.where(x != 0, (x) * np.log10(x / xx), 0)
                pointwise_r[i,:] = np.mean(tmp_pw,axis=0)
            else:
                tmp = np.zeros([self.window]) + self.weight
                pointwise_r[i,:] = np.arange(self.data.shape[1])
            
            T_hat[i] = np.mean(tmp)
            T_sd[i] = np.std(tmp)
                
        self.tsignal = T_hat
        self.tsigma = T_sd
        self.tpointwise = pointwise_r

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

        out = {
            'novelty': self.nsignal.tolist(),
            'novelty_sigma': self.nsigma.tolist(),
            'transience': self.tsignal.tolist(),
            'transience_sigma': self.tsigma.tolist(),
            'resonance': self.rsignal.tolist(),
            'resonance_sigma': self.rsigma.tolist(),
            'nslices':self.nslices,
            'tslices':self.tslices,
        }

        self.signals = out

    def fit_save(self, meas, path, slice_w=False):
        with open(path, 'w') as f:
            json.dump(self.signals, f)
