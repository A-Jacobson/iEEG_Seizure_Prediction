import os
import re
import numpy as np
from glob import glob
from scipy.io import loadmat
import random

def get_label(fname):
    split = re.sub('.mat', '', fname).split('_')
    return int(split[2])

def read_file(path):
    try:
        mat = loadmat(path)
        names = mat['dataStruct'].dtype.names
        ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
        data = ndata['data']
        return data.T
    except Exception as e:
        print os.path.basename(path), e

def load_data(dirname, dirpath=os.path.abspath(os.path.join('E:', 'Seizure_Data')), sample=0):
    path = os.path.join(dirpath, dirname, '*.mat')
    files = glob(path)
    if sample > 0:
        files = random.sample(files, sample)
    print "loading %d files" % (len(files))
    X = (read_file(f) for f in files)
    fnames = map(os.path.basename, files)
    if 'train' in dirname:
        y = np.array(map(get_label, fnames))
        return X, y, fnames
    return X, fnames

def write_prediction(files, preds):
    pass



if __name__=='__main__':
    pass
