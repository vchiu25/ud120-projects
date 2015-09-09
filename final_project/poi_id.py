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
from sklearn.feature_selection import RFECV, SelectKBest, SelectPercentile, f_classif
from sklearn.preprocessing import Imputer, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from feature_format import featureFormat, targetFeatureSplit



def features_select(features_list):
    enron_data_dict = pickle.load(open("final_project_dataset.pkl", "r"))
    enron_data_dict.pop('TOTAL')
    my_dataset = enron_data_dict
    data = featureFormat(my_dataset, features_list, remove_NaN=True, sort_keys = False)
    labels, features = targetFeatureSplit(data)

    for k in [5,8,10,15]:
        selector = SelectKBest(k=k)
        selector.fit_transform(features, labels)
        print k
        print selector.pvalues_
        print selector.get_support()
        for idx, val in enumerate(selector.get_support()):
            if val == True:
                print features_list[1:][idx]
                print selector.pvalues_[idx]
        


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = [
        'poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
        'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
        'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',
        'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages',
        'from_this_person_to_poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
enron_data_dict = pickle.load(open("final_project_dataset.pkl", "r"))

### Task 2: Remove outliers
def remove_outliers():
    enron_data_dict['BANNANTINE JAMES M']['total_payments'] = 56301
    enron_data_dict['BANNANTINE JAMES M']['deferred_income'] = 'NaN'
    enron_data_dict['BANNANTINE JAMES M']['other'] = 'NaN' 

    enron_data_dict['BHATNAGAR SANJAY']['total_payments'] = 137864
    enron_data_dict['BHATNAGAR SANJAY']['expenses'] = 137864
    enron_data_dict['BHATNAGAR SANJAY']['other'] = 'NaN'
    enron_data_dict['BHATNAGAR SANJAY']['director_fees'] = 'NaN'
    enron_data_dict['BHATNAGAR SANJAY']['exercised_stock_options'] = 15456290
    enron_data_dict['BHATNAGAR SANJAY']['restricted_stock'] = 2604490
    enron_data_dict['BHATNAGAR SANJAY']['restricted_stock_deferred'] = -2604490
    enron_data_dict['BHATNAGAR SANJAY']['total_stock_value'] = 15456290

    enron_data_dict['BELFER ROBERT']['total_stock_value'] = 'NaN'
    enron_data_dict['BELFER ROBERT']['restricted_stock_deferred'] = -44093
    enron_data_dict['BELFER ROBERT']['restricted_stock'] = 44093
    enron_data_dict['BELFER ROBERT']['exercised_stock_options'] = 'NaN'
    enron_data_dict['BELFER ROBERT']['total_payments'] = 3285
    enron_data_dict['BELFER ROBERT']['director_fees'] = 102500
    enron_data_dict['BELFER ROBERT']['expenses'] = 3285
    enron_data_dict['BELFER ROBERT']['other'] = 'NaN'
    enron_data_dict['BELFER ROBERT']['deferral_payments'] = 'NaN'
    enron_data_dict['BELFER ROBERT']['deferred_income'] = 'NaN'
    enron_data_dict['BELFER ROBERT']['long_term_incentive'] = -102500

    enron_data_dict.pop('TOTAL')
remove_outliers()

### Task 3: Create new feature(s)

# Define a list of ratio to create new features. Name of the feature will be the first element + ratio. 
# Ratio will be defined as first element / second element in the tuple. 
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

#run kselect

features_select([
        'poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
        'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
        'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',
        'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages',
        'from_this_person_to_poi', 'shared_receipt_with_poi', 'to_message_ratio', 'from_messages_ratio'])

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
#Helper function for parameter tuning
def parameter_tuning(tuned_parameters):
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    test_clf = GridSearchCV(clf, tuned_parameters, cv=StratifiedShuffleSplit(labels, n_iter=100), verbose=10, scoring='f1')
    test_clf.fit(features, labels)
    print(test_clf.best_params_)
    print(test_clf.grid_scores_)
    print("Best parameters set found on development set:")
    print()
    print(test_clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()

    for params, mean_score, scores in test_clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
    print()

# Function to test classifier
test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)

