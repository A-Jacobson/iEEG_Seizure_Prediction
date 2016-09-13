from scipy.signal import butter, lfilter, resample, firwin, decimate
from sklearn.decomposition import FastICA, PCA
import numpy as np


class FFT:
    """
    Apply Fast Fourier Transform to the last axis.
    """
    def get_name(self):
        return "fft"

    def apply(self, data):
        axis = data.ndim - 1
        return np.fft.rfft(data, axis=axis)


class ICA:
    """
    apply ICA experimental!
    """
    def __init__(self, n_components=None):
        self.n_components = n_components

    def get_name(self):
        if self.n_components != None:
            return "ICA%d" % (self.n_components)
        else:
            return 'ICA'

    def apply(self, data):
        # apply pca to each
        ica = FastICA()
        data = ica.fit_transform(da)
        return data


class Resample:
    """
    Resample time-series data.
    """

    def __init__(self, sample_rate):
        self.f = sample_rate

    def get_name(self):
        return "resample%d" % self.f

    def apply(self, data):
        axis = data.ndim - 1
        if data.shape[-1] > self.f:
            return resample(data, self.f, axis=axis)
        return data


class Magnitude:
    """
    Take magnitudes of Complex data
    """
    def get_name(self):
        return "mag"

    def apply(self, data):
        return np.absolute(data)


class LPF:
    """
    Low-pass filter using FIR window
    """

    def __init__(self, f):
        self.f = f

    def get_name(self):
        return 'lpf%d' % self.f

    def apply(self, data):
        nyq = self.f / 2.0
        cutoff = min(self.f, nyq - 1)
        h = firwin(numtaps=101, cutoff=cutoff, nyq=nyq)
        # data[ch][dim0]
        # apply filter over each channel
        for j in range(len(data)):
            data[j] = lfilter(h, 1.0, data[j])

        return data


class Mean:
    """
    extract channel means
    """

    def get_name(self):
        return 'mean'

    def apply(self, data):
        axis = data.ndim - 1
        return data.mean(axis=axis)


class Abs:
    """
    extract channel means
    """

    def get_name(self):
        return 'abs'

    def apply(self, data):
        return np.abs(data)


class Stats:
    """
    Subtract the mean, then take (min, max, standard_deviation) for each channel.
    """
    def get_name(self):
        return "stats"

    def apply(self, data):
        # data[ch][dim]
        shape = data.shape
        out = np.empty((shape[0], 3))
        for i in range(len(data)):
            ch_data = data[i]
            ch_data = data[i] - np.mean(ch_data)
            outi = out[i]
            outi[0] = np.std(ch_data)
            outi[1] = np.min(ch_data)
            outi[2] = np.max(ch_data)
        return out


class Log10:
    """
    Apply Log10
    """

    def get_name(self):
        return "log10"

    def apply(self, data):
        # interps 0 data before taking log
        indices = np.where(data <= 0)
        data[indices] = np.max(data)
        data[indices] = (np.min(data) * 0.1)
        return np.log10(data)


class Slice:
    """
    Take a slice of the data on the last axis.
    e.g. Slice(1, 48) works like a normal python slice, that is 1-47 will be taken
    """

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def get_name(self):
        return "slice%d-%d" % (self.start, self.end)

    def apply(self, data):
        s = [slice(None), ] * data.ndim
        s[-1] = slice(self.start, self.end)
        return data[s]


class CorrelationMatrix:
    """
    Calculate correlation coefficients matrix across all EEG channels.
    """
    def get_name(self):
        return 'corr-mat'

    def apply(self, data):
        return upper_right_triangle(np.corrcoef(data))


class Eigenvalues:
    """
    Take eigenvalues of a matrix, and sort them by magnitude in order to
    make them useful as features (as they have no inherent order).
    """
    def get_name(self):
        return 'eigenvalues'

    def apply(self, data):
        w, v = np.linalg.eig(data)
        w = np.absolute(w)
        w.sort()
        return w

def upper_right_triangle(matrix):
    indices = np.triu_indices_from(matrix)
    return np.asarray(matrix[indices])

def slidingWindow(sequence, window, step=1):
    for i in range(0, sequence.shape[1] - window + 1, step):
        yield sequence[:, i:i + window]


def decimate(time_data):
    '''decimates by factor of 10'''
    return decimate(time_data, 10, 'fir', axis=1)
