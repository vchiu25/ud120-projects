#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
import matplotlib.pyplot as plt
import operator
import pickle
import sys
from sklearn.cross_validation import train_test_split, KFold, StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import Imputer, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

#outlier cleaning in general
#user best k to select feature
#run algorithm

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'from_poi_ratio', 'to_poi_ratio', 'exercised_stock_ratio']
features_list = ['salary', 'to_messages', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'total_stock_value', 'shared_receipt_with_poi', 'long_term_incentive', 'exercised_stock_options', 'from_messages', 'other', 'from_poi_to_this_person', 'from_this_person_to_poi', 'poi', 'deferred_income', 'expenses', 'restricted_stock', 'director_fees']

### Load the dictionary containing the dataset
enron_data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

data = featureFormat(enron_data_dict, features_list, remove_NaN=True, sort_keys = False)
labels, features = targetFeatureSplit(data)
'''
imp = Imputer(missing_values='NaN', strategy='mean', copy=True)
features = imp.fit_transform(features)
'''
clf = DecisionTreeClassifier()
clf.fit(features, labels)
feature_importance = clf.feature_importances_
print feature_importance
print sorted(enumerate(feature_importance ), key=operator.itemgetter(1), reverse=True)
print features_list[11], features_list[6], features_list[4], features_list[0]
print max(enumerate(feature_importance ), key=operator.itemgetter(1))
