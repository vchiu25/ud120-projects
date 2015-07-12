#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
import numpy as np
import operator
import pickle
from scipy.stats import percentileofscore
from sets import Set
import sys
from sklearn.cross_validation import train_test_split, KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_selection import f_classif, SelectKBest, SelectPercentile, VarianceThreshold, RFE, RFECV
from sklearn.preprocessing import Imputer, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 
		'total_stock_value', 'shared_receipt_with_poi', 'long_term_incentive', 'exercised_stock_options', 'from_messages', 'other', 
		'from_poi_to_this_person', 'from_this_person_to_poi', 'deferred_income', 'expenses', 'restricted_stock', 'director_fees']

### Load the dictionary containing the dataset
enron_data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers
for outlier in ['BELFER ROBERT', 'BHATNAGAR SANJAY', 'KAMINSKI WINCENTY J', 'TOTAL', 'BANNANTINE JAMES M']:
    enron_data_dict.pop(outlier)

def check_NaN(data, feature_tuple):
    if data[feature_tuple[0]] == 'NaN' or data[feature_tuple[1]] == 'NaN' or data[feature_tuple[1]] == 0:
        return False
    return True

def set_ratio(person, data, feature_tuple):
    if check_NaN(data, feature_tuple):
        enron_data_dict[person][feature_tuple[0] + '_ratio'] = float(data[feature_tuple[0]]) / float(data[feature_tuple[1]])
    else:
        enron_data_dict[person][feature_tuple[0] + '_ratio'] = 'NaN'

new_features = [
    ('to_messages', 'from_poi_to_this_person'), 
    ('from_messages', 'from_this_person_to_poi'),
    ('exercised_stock_options', 'salary'), 
    ('salary', 'total_payments'), 
    ('bonus', 'total_payments'),
    ('long_term_incentive', 'total_payments'), 
    ('expenses', 'total_payments'),
    ('total_stock_value', 'exercised_stock_options'), 
    ('restricted_stock', 'total_stock_value')
]

### Task 3: Create new feature(s)
for person, data in enron_data_dict.items():
    for new_feature in new_features:
        set_ratio(person, data, new_feature)

features_list.extend([new_feature[0] + '_ratio' for new_feature in new_features])
### Store to my_dataset for easy export below.
my_dataset = enron_data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, remove_NaN=True, sort_keys = False)
print data
labels, features = targetFeatureSplit(data)

# Recursive feature elimination
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from sklearn.feature_selection import RFECV, RFE
from sklearn.datasets import make_classification

# Build a classification task using 3 informative features
X, y = make_classification(n_samples=1000, n_features=25, n_informative=3,
                           n_redundant=2, n_repeated=0, n_classes=8,
                           n_clusters_per_class=1, random_state=0)

# Create the RFE object and compute a cross-validated score.

# The "accuracy" scoring is proportional to the number of correct
# classifications
print 'start fitting'
svc = SVC(kernel='linear', C=1000)
scaler = MinMaxScaler()
features = scaler.fit_transform(features)
rfe = RFECV(estimator=DecisionTreeClassifier(), step=1, cv=StratifiedShuffleSplit(labels, n_iter=500), scoring='f1', verbose=10)
rfe.fit(features, labels)

print("Optimal number of features : %d" % rfe.n_features_)
print("Support : %s" % rfe.support_)
print("Ranking : %s" % rfe.ranking_)
list = []

#['salary', 'to_messages', 'deferral_payments', 'total_payments', 'loan_advances', 'restricted_stock_deferred', 'total_stock_value', 'shared_receipt_with_poi', 'long_term_incentive', 'exercised_stock_options', 'from_messages', 'other', 'from_poi_to_this_person', 'from_this_person_to_poi', 'deferred_income', 'expenses', 'restrited_stock', 'to_poi_ratio', 'stock_ratio']
# ['salary', 'deferral_payments', 'loan_advances', 'restricted_stock_deferred', 'total_stock_value', 'long_term_incentive', 'exercised_stock_options', 'other', 'deferred_income', 'to_poi_ratio']
for idx, val in enumerate(rfe.support_):
    if val == True:
        list.append(features_list[idx+1])
print list
