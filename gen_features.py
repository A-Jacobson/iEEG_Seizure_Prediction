from pipeline import Pipeline
from eeg_io import load_data
from transforms import *
import multiprocessing as mp

pipelines = [
    # statstical features
    Pipeline([Mean()]),
    Pipeline([Mean(), Abs()]),
    Pipeline([Stats()]),
    Pipeline([CorrelationMatrix()]),
    # Pipeline([CorrelationMatrix(), Eigenvalues()]), # under construction

    # time domain features
    Pipeline([Resample(600)]),
    Pipeline([LPF(5.0), Resample(600)]),
    Pipeline([Interp(), Resample(600)]),
    Pipeline([Resample(1200)]),
    Pipeline([LPF(5.0), Resample(1200)]),
    Pipeline([Interp(), Resample(1200)]),
    # frequency domain features
    Pipeline([FFT(), Slice(1, 48), Magnitude(), Log10()]),
    Pipeline([FFT(), Slice(1, 64), Magnitude(), Log10()]),
    Pipeline([FFT(), Slice(1, 96), Magnitude(), Log10()]),
    Pipeline([FFT(), Slice(1, 128), Magnitude(), Log10()]),
    Pipeline([FFT(), Slice(1, 160), Magnitude(), Log10()]),
    # combination features (under construction)
    # Pipeline([FFTWithTimeFreqCorrelation(1, 48, 400, 'usf')]), 
    # Pipeline([FFTWithTimeFreqCorrelation(1, 48, 400, 'usf')]),
]

folders = ['train_1', 'test_1', 'train_2', 'test_2', 'train_3', 'test_3']

def gen_features(folder):
    if 'train' in folder:
        for p in pipelines:
            X, y, files = load_data(folder)
            p.to_file(X, files, folder, y)
    else:
        for p in pipelines:
            X, files = load_data(folder)
            p.to_file(X, files, folder)


if __name__ == '__main__':
    p = mp.Pool(6)
    p.map(gen_features, folders)
