import pandas as pd
from scipy.io import loadmat
from scipy import signal
import os
import re
from glob import glob
import numpy as np

class IEEG_Data:

    def __init__(self, dirpath=os.path.abspath(os.path.join('E:', 'Seizure_Data'))):
        self.dirpath = dirpath

    def preprocess(self, x):
        return signal.resample(x, 600, window=400).flatten('F')

    def get_label(self, path):
        '''
        splits file name into patient, sample, label
        '''
        fname = os.path.basename(path)
        split = re.sub('.mat', '', fname).split('_')
        return split

    def read_file(self, path, train=True):
        mat = loadmat(path)
        names = mat['dataStruct'].dtype.names
        ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
        data = ndata['data']
        if train:
            _, _, label = self.get_label(path)
            return data, label
        else:
            return data

    def load(self, dirname):
        path = os.path.join(self.dirpath, dirname, '*.mat')
        files = glob(path)
        if 'train' in dirname:
            X = []
            y = []
            i = 1
            for f in files:
                if i % 100 == 0:
                    print "loading file ", i, " of ", len(files)
                try:
                    x_i, y_i = self.read_file(f)
                    x_i = self.preprocess(x_i)
                    X.append(x_i)
                    y.append(y_i)
                    i += 1
                except Exception as e:
                    print f
                    print e
            return np.array(X), np.array(y), map(os.path.basename, files)
        else:
            for f in files:
                if i % 100 == 0:
                    print "loading file ", i, " of ", len(files)
                try:
                    x_i = self.read_file(f, train=False)
                    x_i = self.preprocess(x_i)
                    X.append(x_i)
                    i += 1
                except Exception as e:
                    print f
                    print e
            return np.array(X), map(os.path.basename, files)

if __name__ == '__main__':
