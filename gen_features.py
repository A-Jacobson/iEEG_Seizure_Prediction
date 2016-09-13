from pipeline import Pipeline
from data_utils import load_data
from transforms import *


pipelines = [
    Pipeline([Mean()]), # statstical features
    Pipeline([Mean(), Abs()]),
    Pipeline([Stats()]),
    Pipeline([Resample(600)]), # time domain features
    Pipeline([LPF(), Resample(600)]),
    Pipeline([FFT(), Slice(1, 48), Magnitude(), Log10()]), # frequency domain features
    Pipeline([FFT(), Slice(1, 64), Magnitude(), Log10()]),
    Pipeline([FFT(), Slice(1, 96), Magnitude(), Log10()]),
    Pipeline([FFT(), Slice(1, 128), Magnitude(), Log10()]),
    Pipeline([FFT(), Slice(1, 160), Magnitude(), Log10()]),
    Pipeline([FFT(), Magnitude(), Log10()])
    ]

train_folders = ['train_1', 'train_2', 'train_3']
test_folders = ['test_1', 'test_2', 'test_3']

for t in train_folders:
    for p in pipelines:
        X, y, files = load_data(t)
        p.to_file(X, files, t, y)

for t in test_folders:
    for p in pipelines:
        X, files = load_data(t)
        p.to_file(X, files, t)
