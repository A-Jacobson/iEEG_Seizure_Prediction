import os
import re
import numpy as np
from glob import glob
from scipy.io import loadmat
import random
from gen_features import extract_features
import shutil


def get_skipped(dirname, dirpath=os.path.abspath(os.path.join('E:', 'Seizure_Data', 'skip'))):
    path = os.path.join(dirpath, dirname, '*.mat')
    files = np.array(map(os.path.basename, glob(path)))
    preds = np.zeros(len(files))
    return preds, files


def get_label(fname):
    split = re.sub('.mat', '', fname).split('_')
    return int(split[2])


def read_file(path, features=False):
    try:
        mat = loadmat(path, verify_compressed_data_integrity=False)
        names = mat['dataStruct'].dtype.names
        ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
        data = ndata['data']
        if features == True:
            return extract_features(data.T)
        else:
            return data.T
    except Exception as e:
        print os.path.basename(path), e


def remove_full_dropout(dirname, dirpath=os.path.abspath(os.path.join('E:', 'Seizure_Data'))):
    path = os.path.join(dirpath, dirname, '*.mat')
    dest = os.path.join(dirpath, 'skip', dirname)
    if not os.path.exists(dest):
        os.makedirs(dest)
    files = glob(path)
    for f in files:
        if read_file(f).std() == 0:
            print 'moving %s to %s' % (f, dest)
            shutil.move(f, dest)


def load_data(dirname, dirpath=os.path.abspath(os.path.join('E:', 'Seizure_Data')), sample=0, features=False):
    path = os.path.join(dirpath, dirname, '*.mat')
    files = glob(path)
    if sample > 0:
        files = random.sample(files, sample)

    print "loading %d files" % (len(files))
    fnames = np.array(map(os.path.basename, files))
    if features == True:
        X = (read_file(f, features=True) for f in files)
    else:
        X = (read_file(f, features=False) for f in files)

    if 'train' in dirname:
        y = np.array(map(get_label, fnames))
        return X, y, fnames
    return X, fnames


def preds_to_df(files, preds):
    return pd.DataFrame({'File': files, 'Class': preds})


if __name__ == '__main__':
    pass
