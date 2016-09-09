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

print "loading X_1"
X_1 = np.array(list(X_1))
print "resampling X_1"
X_1_sm, y_1_sm = sm.fit_sample(X_1, y_1)

print "loading X_2"
X_2 = np.array(list(X_2))
print "resampling X_2"
X_2_sm, y_2_sm = sm.fit_sample(X_2, y_2)

print "loading X_3"
X_3 = np.array(list(X_3))
print "resampling X_3"
X_3_sm, y_3_sm = sm.fit_sample(X_3, y_3)


Ada_1 = AdaBoostClassifier(n_estimators=100, random_state=1337)
Ada_2 = AdaBoostClassifier(n_estimators=100, random_state=1337)
Ada_3 = AdaBoostClassifier(n_estimators=100, random_state=1337)

print "Fit models"
Ada_1.fit(X_1_sm, y_1_sm)
Ada_2.fit(X_2_sm, y_2_sm)
Ada_3.fit(X_3_sm, y_3_sm)

print "Saving models"
cPickle.dump(Ada_1, 'Ada_1.pkl')
cPickle.dump(Ada_2, 'Ada_2.pkl')
cPickle.dump(Ada_3, 'Ada_3.pkl')
