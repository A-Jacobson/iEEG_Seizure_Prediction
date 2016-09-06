import pandas as pd
from scipy.io import loadmat
from scipy import signal
import os
import re
from glob import glob
import numpy as np

def preprocess(sample):
    return signal.resample(sample, 600, window=400).flatten('F')

class IEEG_Data:

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

    def read_files(self, dirpath=os.path.abspath(os.path.join('E:', 'Seizure_Data', 'train_1', '*.mat'))):
        files = glob(dirpath)
        if 'train' in dirpath:
            X = []
            y = []
            for f in files:
                i = 1
                if i % 100 == 0:
                    print "file ", i, " of ", len(files)
                    print os.path.basename(f)
                x_i, y_i = self.read_file(f)
                x_i = self.preprocess(x_i)
                X.append(x_i)
                y.append(y_i)
                i += 1
            return np.array(X), np.array(y)
        else:
            X = np.array([self.preprocess(self.read_file(f, train=False)) for f in files])
            return X
