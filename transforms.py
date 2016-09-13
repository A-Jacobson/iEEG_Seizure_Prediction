from scipy.signal import butter, lfilter, resample, firwin, decimate
from sklearn.decomposition import FastICA, PCA
from sklearn import preprocessing
import numpy as np


class UnitScale:
    """
    Scale across the last axis.
    """

    def get_name(self):
        return 'unit-scale'

    def apply(self, data):
        return preprocessing.scale(data, axis=data.ndim - 1)


class UnitScaleFeat:
    """
    Scale across the first axis, i.e. scale each feature.
    """

    def get_name(self):
        return 'unit-scale-feat'

    def apply(self, data):
        return preprocessing.scale(data, axis=0)


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


class Interp:
    """
    Interpolate zeros max --> min * 1.0
    NOTE: try different methods later
    """

    def get_name(self):
        return "interp"

    def apply(self, data):
        # interps 0 data before taking log
        indices = np.where(data <= 0)
        data[indices] = np.max(data)
        data[indices] = (np.min(data) * 0.1)
        return data


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


# Fix everything below here

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


class FreqCorrelation:
    """
    Correlation in the frequency domain. First take FFT with (start, end) slice options,
    then calculate correlation co-efficients on the FFT output, followed by calculating
    eigenvalues on the correlation co-efficients matrix.
    The output features are (fft, upper_right_diagonal(correlation_coefficients), eigenvalues)
    Features can be selected/omitted using the constructor arguments.
    """

    def __init__(self, start, end, scale_option, with_fft=False, with_corr=True, with_eigen=True):
        self.start = start
        self.end = end
        self.scale_option = scale_option
        self.with_fft = with_fft
        self.with_corr = with_corr
        self.with_eigen = with_eigen
        assert scale_option in ('us', 'usf', 'none')
        assert with_corr or with_eigen

    def get_name(self):
        selections = []
        if not self.with_corr:
            selections.append('nocorr')
        if not self.with_eigen:
            selections.append('noeig')
        if len(selections) > 0:
            selection_str = '-' + '-'.join(selections)
        else:
            selection_str = ''
        return 'freq-correlation-%d-%d-%s-%s%s' % (self.start, self.end, 'withfft' if self.with_fft else 'nofft',
                                                   self.scale_option, selection_str)

    def apply(self, data):
        data1 = FFT().apply(data)
        data1 = Slice(self.start, self.end).apply(data1)
        data1 = Magnitude().apply(data1)
        data1 = Log10().apply(data1)

        data2 = data1
        if self.scale_option == 'usf':
            data2 = UnitScaleFeat().apply(data2)
        elif self.scale_option == 'us':
            data2 = UnitScale().apply(data2)

        data2 = CorrelationMatrix().apply(data2)

        if self.with_eigen:
            w = Eigenvalues().apply(data2)

        out = []
        if self.with_corr:
            data2 = upper_right_triangle(data2)
            out.append(data2)
        if self.with_eigen:
            out.append(w)
        if self.with_fft:
            data1 = data1.ravel()
            out.append(data1)
        for d in out:
            assert d.ndim == 1

        return np.concatenate(out, axis=0)


class TimeCorrelation:
    """
    Correlation in the time domain. First downsample the data, then calculate correlation co-efficients
    followed by calculating eigenvalues on the correlation co-efficients matrix.
    The output features are (upper_right_diagonal(correlation_coefficients), eigenvalues)
    Features can be selected/omitted using the constructor arguments.
    """

    def __init__(self, max_hz, scale_option, with_corr=True, with_eigen=True):
        self.max_hz = max_hz
        self.scale_option = scale_option
        self.with_corr = with_corr
        self.with_eigen = with_eigen
        assert scale_option in ('us', 'usf', 'none')
        assert with_corr or with_eigen

    def get_name(self):
        selections = []
        if not self.with_corr:
            selections.append('nocorr')
        if not self.with_eigen:
            selections.append('noeig')
        if len(selections) > 0:
            selection_str = '-' + '-'.join(selections)
        else:
            selection_str = ''
        return 'time-correlation-r%d-%s%s' % (self.max_hz, self.scale_option, selection_str)

    def apply(self, data):
        # so that correlation matrix calculation doesn't crash
        for ch in data:
            if np.alltrue(ch == 0.0):
                ch[-1] += 0.00001

        data1 = data
        if data1.shape[1] > self.max_hz:
            data1 = Resample(self.max_hz).apply(data1)

        if self.scale_option == 'usf':
            data1 = UnitScaleFeat().apply(data1)
        elif self.scale_option == 'us':
            data1 = UnitScale().apply(data1)

        data1 = CorrelationMatrix().apply(data1)

        if self.with_eigen:
            w = Eigenvalues().apply(data1)

        out = []
        if self.with_corr:
            data1 = upper_right_triangle(data1)
            out.append(data1)
        if self.with_eigen:
            out.append(w)

        for d in out:
            assert d.ndim == 1

        return np.concatenate(out, axis=0)


class TimeFreqCorrelation:
    """
    Combines time and frequency correlation, taking both correlation coefficients and eigenvalues.
    """

    def __init__(self, start, end, max_hz, scale_option):
        self.start = start
        self.end = end
        self.max_hz = max_hz
        self.scale_option = scale_option
        assert scale_option in ('us', 'usf', 'none')

    def get_name(self):
        return 'time-freq-correlation-%d-%d-r%d-%s' % (self.start, self.end, self.max_hz, self.scale_option)

    def apply(self, data):
        data1 = TimeCorrelation(self.max_hz, self.scale_option).apply(data)
        data2 = FreqCorrelation(self.start, self.end,
                                self.scale_option).apply(data)
        assert data1.ndim == data2.ndim
        return np.concatenate((data1, data2), axis=data1.ndim - 1)


class FFTWithTimeFreqCorrelation:
    """
    Combines FFT with time and frequency correlation, taking both correlation coefficients and eigenvalues.
    """

    def __init__(self, start, end, max_hz, scale_option):
        self.start = start
        self.end = end
        self.max_hz = max_hz
        self.scale_option = scale_option

    def get_name(self):
        return 'fft-with-time-freq-corr-%d-%d-r%d-%s' % (self.start, self.end, self.max_hz, self.scale_option)

    def apply(self, data):
        data1 = TimeCorrelation(self.max_hz, self.scale_option).apply(data)
        data2 = FreqCorrelation(self.start, self.end,
                                self.scale_option, with_fft=True).apply(data)
        assert data1.ndim == data2.ndim
        return np.concatenate((data1, data2), axis=data1.ndim - 1)


def upper_right_triangle(matrix):
    indices = np.triu_indices_from(matrix)
    return np.asarray(matrix[indices])


def slidingWindow(sequence, window, step=1):
    for i in range(0, sequence.shape[1] - window + 1, step):
        yield sequence[:, i:i + window]


def decimate(time_data):
    '''decimates by factor of 10'''
    return decimate(time_data, 10, 'fir', axis=1)
