# iEEG Seizure Prediction
Seizure prediction model from Kaggle's iEEG Data. For https://www.kaggle.com/c/melbourne-university-seizure-prediction

## Workflow

- load .mat files
- check class balance

## Preprocessing
- Data consists of 10 minute buffers of multichannel iEEG signal data, split into two classes (Interictal and Preictal)
- Use Signal processing techniques to extract relevant features from data

## Relevant Resources

- Prediction of the onset of epileptic seizures from iEEG data: http://cs229.stanford.edu/proj2014/Shima%20Alizadeh,%20Scott%20Davidson,%20Ari%20Frankel,Prediction%20Onset%20Epileptic.pdf
- 2014 Kaggle competition with Dog iEEG's: https://www.kaggle.com/c/seizure-prediction
- https://github.com/MichaelHills/seizure-detection
- Signal Processing for Machine Learning: https://www.youtube.com/watch?v=VO0d6EuGpO0
