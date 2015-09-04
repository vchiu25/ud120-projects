#!/usr/bin/python

import sys
import numpy as np
import operator
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from scipy.stats import percentileofscore
from sets import Set
from sklearn.cross_validation import train_test_split, KFold, StratifiedShuffleSplit
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV, ParameterGrid
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import Imputer, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from feature_format import featureFormat, targetFeatureSplit

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

#Define features_list
features_list = ['poi', 'salary', 'deferral_payments', 'loan_advances', 'bonus', 'shared_receipt_with_poi', 'long_term_incentive', 'exercised_stock_options', 'from_messages', 'from_this_person_to_poi', 'deferred_income', 'to_messages_ratio', 'salary_ratio']

### Load the dictionary containing the dataset
enron_data_dict = pickle.load(open("final_project_dataset.pkl", "r"))

### Task 2: Remove outliers
for test in ['THE TRAVEL AGENCY IN THE PARK']:
    print enron_data_dict[test]

for outlier in ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']:
    enron_data_dict.pop(outlier)

### Task 3: Create new feature(s)

# Define a list of ratio to create new features. Name of the feature will be the first element + ratio. 
# Ratio will be defined as first element / second element in the tuple. 
new_features = [
    ('to_messages', 'from_poi_to_this_person'),
    ('from_messages', 'from_this_person_to_poi'),
    ('salary', 'total_payments'),
]

# Helper function for set_ratio. Check to make sure the two values is not NaN denominator  value is not 0
def check_NaN(data, feature_tuple):
    if data[feature_tuple[0]] == 'NaN' or data[feature_tuple[1]] == 'NaN' or data[feature_tuple[1]] == 0:
        return False
    return True

# Calculate and set the ratio for the enron_data to create new feature
def set_ratio(person, data, feature_tuple):
    if check_NaN(data, feature_tuple):
        enron_data_dict[person][feature_tuple[0] + '_ratio'] = float(data[feature_tuple[0]]) / float(data[feature_tuple[1]])
    else:
        enron_data_dict[person][feature_tuple[0] + '_ratio'] = 'NaN'

# Create new feature
for person, data in enron_data_dict.items():
    for new_feature in new_features:
        set_ratio(person, data, new_feature)

### Store to my_dataset for easy export below.
my_dataset = enron_data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, remove_NaN=True, sort_keys = False)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Create classifier pipeline with minmaxscaler processing
# Classifier definition with tuned parameter
svm_clf = SVC(kernel='linear', C=1000)
dtree_clf = DecisionTreeClassifier(criterion='gini', min_samples_split=10, max_features='sqrt', min_samples_leaf=1)
adab_clf = AdaBoostClassifier(n_estimators=200, learning_rate=0.5) 
# Pipeliner defintion
clf = Pipeline([('scaler', MinMaxScaler()), ('classifier', svm_clf)])

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


# Function to test classifier
test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)
