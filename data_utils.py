import pandas as pd
from scipy.io import loadmat
from scipy import signal
import os
import re
from glob import glob
import numpy as np
import multiprocessing as mp


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def resample(x):
    return signal.resample(x, 600, window=400).astype("float32").flatten('F')

def get_label(path):
    fname = os.path.basename(path)
    split = re.sub('.mat', '', fname).split('_')
    return int(split[2])

def read_file(path):
    try:
        mat = loadmat(path)
        names = mat['dataStruct'].dtype.names
        ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
        data = ndata['data']
        return data
    except Exception as e:
        print os.path.basename(path)
        print e

def load_data(dirname, shape="flat", processes=mp.cpu_count()-1, dirpath=os.path.abspath(os.path.join('E:', 'Seizure_Data')), sample=False):
    path = os.path.join(dirpath, dirname, '*.mat')
    files = glob(path)
    if sample == True:
        files = files[0:20]
    print "loading", len(files), "with", processes, "processes."
    p = mp.Pool(processes)
    X = np.array(p.map(read_file, files))
    files = np.array(p.map(os.path.basename, files))
    if shape == 'cnn':
        X = X.reshape((len(X), 16, 600, 1))  # same as n x channels x w x h
    if 'train' in dirname:
        y = np.array(p.map(get_label, files))
        return X, y, files
    return X, files




if __name__=='__main__':
    pass
