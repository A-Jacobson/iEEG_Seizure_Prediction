from data_utils import load_data
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.cross_validation import train_test_split
from imblearn.over_sampling import SMOTE
import numpy as np
import cPickle


X_1, y_1, _ = load_data('train_1', features=True)
X_2, y_2, _ = load_data('train_2', features=True)
X_3, y_3, _ = load_data('train_3', features=True)

sm = SMOTE(kind='regular')
Ada_1 = AdaBoostClassifier(n_estimators=100, random_state=1337)
Ada_2 = AdaBoostClassifier(n_estimators=100, random_state=1337)
Ada_3 = AdaBoostClassifier(n_estimators=100, random_state=1337)

# load, smote, fit, save
X_1 = np.array(list(X_1))
X_1_sm, y_1_sm = sm.fit_sample(X_1, y_1)
Ada_1.fit(X_1_sm, y_1_sm)
with open('Ada_1.pkl', 'wb') as f:
    cPickle.dump(Ada_1, f)

X_2 = np.array(list(X_2))
X_2_sm, y_2_sm = sm.fit_sample(X_2, y_2)
Ada_2.fit(X_2_sm, y_2_sm)
with open('Ada_2.pkl', 'wb') as f:
    cPickle.dump(Ada_2, f)

X_3 = np.array(list(X_3))
X_3_sm, y_3_sm = sm.fit_sample(X_3, y_3)
Ada_3.fit(X_3_sm, y_3_sm)
with open('Ada_3.pkl', 'wb') as f:
    cPickle.dump(Ada_3, f)
