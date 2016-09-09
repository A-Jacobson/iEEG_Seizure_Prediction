# iEEG Seizure Prediction

Seizure prediction model from Kaggle's iEEG Data. For <https://www.kaggle.com/c/melbourne-university-seizure-prediction>

## General Strategy

- build 3, per subject classifiers.
- format multichannel signal chunks as images and feed time domain features to model.
- extract featurevector using signal processing techniques such as frequency domain analysis and feed to multiple non-deep classifiers.
- feed featurevector generator by cnn to other classifiers.
- try different methods such as smote and cost function weighting to combat class imbalance.
- ensemble.
- try meta classifiers train on all the data.

## Preprocessing Strategy

- Data consists of 10 minute buffers of multichannel iEEG signal data, split into two classes (Interictal and Preictal).
- Drop full 0 samples.
- filter or interpolate partial 0 samples.
- experiment with statistical features such as `mean`, `std`, and `absmean` frequency per channel.
- resample time domain to 600 samples.
- try different lowpass and bandpass filters.
- try sliding window analysis.
- convert to frequency domain with `rfft`.
- experiment with different fft frequency slices.
- experiment with `absolute` and `log10` of frequency slices to represent magnitude and normally distributed magnitude respectively.
- try wavelet transforms.

## Todo

- refactor code into pipeline for fast iteration with multiple feature transforms/models.
- save different feature map combinations as numpy arrays.
- get xgboost working on windows.

## Relevant Resources

- Prediction of the onset of epileptic seizures from iEEG data: <http://cs229.stanford.edu/proj2014/Shima%20Alizadeh,%20Scott%20Davidson,%20Ari%20Frankel,Prediction%20Onset%20Epileptic.pdf>
- 2014 Kaggle competition with Dog iEEG's: <https://www.kaggle.com/c/seizure-prediction>
- <https://github.com/MichaelHills/seizure-detection>
- Signal Processing for Machine Learning: <https://www.youtube.com/watch?v=VO0d6EuGpO0>
- Time Series Shapelets: <http://delivery.acm.org/10.1145/1560000/1557122/p947-ye.pdf?ip=128.227.226.41&id=1557122&acc=ACTIVE%20SERVICE&key=5CC3CBFF4617FD07%2EC2A817F22E85290F%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&CFID=664809632&CFTOKEN=980235&__acm__=1473189592_e7e9e1b9abf2c9e4f2b992e92661a55b>
- <https://hal.inria.fr/hal-01055103/file/lotte_EEGSignalProcessing.pdf>
- <http://www.mathworks.com/help/signal/examples/practical-introduction-to-frequency-domain-analysis.html>
