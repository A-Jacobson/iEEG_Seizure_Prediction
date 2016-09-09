from scipy.signal import butter, lfilter, resample, firwin, decimate
import sklearn.preprocessing
import numpy as np


def decimate(time_data):
    '''decimates by factor of 10'''
    return decimate(time_data, 10, 'fir', axis=1)


def slidingWindow(sequence, window, step=1):
    for i in range(0, sequence.shape[1] - window + 1, step):
        yield sequence[:, i:i + window]


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a


def firwin_filter(data, f=400):
    nyq = f / 2.0
    cutoff = min(f, nyq - 1)
    h = firwin(numtaps=101, cutoff=cutoff, nyq=nyq)
    return lfilter(h, 1.0, data)


def butter_bandpass_filter(data, lowcut=1.0, highcut=300.0, fs=400.0, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def upper_right_triangle(matrix):
    indices = np.triu_indices_from(matrix)
    return np.asarray(matrix[indices])


def fft(time_data):
    return np.log10(np.absolute(np.fft.rfft(time_data, axis=1)[:, 1:150]))


def time_corr(time_data):
    resampled = resample(time_data, 400, axis=1) \
        if time_data.shape[-1] > 400 else time_data
    scaled = resampled
    scaled = sklearn.preprocessing.scale(resampled, axis=0)
    corr_matrix = np.corrcoef(scaled)
    eigenvalues = np.absolute(np.linalg.eig(corr_matrix)[0])
    corr_coefficients = upper_right_triangle(corr_matrix)
    return np.concatenate((corr_coefficients, eigenvalues))


def freq_corr(fft_data):
    scaled = sklearn.preprocessing.scale(fft_data, axis=0)
    corr_matrix = np.corrcoef(scaled)
    eigenvalues = np.absolute(np.linalg.eig(corr_matrix)[0])
    eigenvalues.sort()
    corr_coefficients = upper_right_triangle(corr_matrix)
    return np.concatenate((corr_coefficients, eigenvalues))


def extract_features(data):
    filtered = firwin_filter(data)
    fft_out = fft(filtered)
    freq_corr_out = freq_corr(fft_out)
    time_corr_out = time_corr(data)
    return np.concatenate((fft_out.ravel(), freq_corr_out, time_corr_out))

# def extract_features(data):
#     filtered = butter_bandpass_filter(data)
#     second_clips = np.array(list(slidingWindow(data,400,400)))
#     return np.array([transform(clip) for clip in second_clips])
