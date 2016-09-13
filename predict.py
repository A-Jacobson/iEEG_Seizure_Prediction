from sklearn.ensemble import AdaBoostClassifier
import cPickle
from data_utils import load_data, preds_to_df, get_skipped
import pandas as pd
import numpy as np


with open('Ada_1.pkl', 'rb') as f:
    Ada_1 = cPickle.load(f)
with open('Ada_3.pkl', 'rb') as f:
    Ada_2 = cPickle.load(f)
with open('Ada_3.pkl', 'rb') as f:
    Ada_3 = cPickle.load(f)

X_1, files_1 = load_data('test_1', features=True)
X_2, files_2 = load_data('test_2', features=True)
X_3, files_3 = load_data('test_3', features=True)

X_1 = np.array(list(X_1))
X_2 = np.array(list(X_2))
X_3 = np.array(list(X_3))

preds_1 = preds_to_df(Ada_1.predict(X_1), files_1)
preds_2 = preds_to_df(Ada_2.predict(X_2), files_2)
preds_3 = preds_to_df(Ada_2.predict(X_3), files_3)

# get preds for skipped examples
skipped_1 = get_skipped('test_1')
skipped_2 = get_skipped('test_2')
skipped_3 = get_skipped('test_3')

submit = pd.concat([preds_1, skipped_1, preds_2,
                    skipped_2, preds_3, skipped_3])

submit = submit[['File', 'Class']]

submit.to_csv('ada100_fft_time_corr.csv',
              index=False)
