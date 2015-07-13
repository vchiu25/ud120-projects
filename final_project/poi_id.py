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

# Helper function to use RFE to select feacture
def feature_selection():
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
    for idx, val in enumerate(rfe.support_):
        if val == True:
            list.append(features_list[idx+1])
    print list

#Define features_list
features_list = ['poi', 'salary', 'deferral_payments', 'loan_advances', 'bonus', 'shared_receipt_with_poi', 'long_term_incentive', 'exercised_stock_options', 'from_messages', 'from_this_person_to_poi', 'deferred_income', 'to_messages_ratio', 'salary_ratio']

### Load the dictionary containing the dataset
enron_data_dict = pickle.load(open("final_project_dataset.pkl", "r"))

### Task 2: Remove outliers

# Helper function for outlier identification
def outliers_identification():
    # Create dictionary of feature. Key = feature name, content = list of (person, value)
    feature_dict = {}
    for person in enron_data_dict.keys():
        for feature in enron_data_dict[person]:
            if feature == 'poi' or feature == 'email_address' :
                continue
            if feature not in feature_dict:
                feature_dict[feature] = []
            if enron_data_dict[person][feature] == 'NaN':
                continue
            else:
                feature_dict[feature].append((person, enron_data_dict[person][feature])) 
    # Sort by value and convert it to percentile
    percentile_threshold = 10. / len(feature_dict)
    outliers = Set([])
    for feature in feature_dict:
        feature_dict[feature].sort(key=lambda x:x[1])
        feautre_value = [x[1] for x in feature_dict[feature]]
        feature_percentiles = [round(percentileofscore(feautre_value, value, 'rank'), 1) for value in feautre_value]
        for index, feature_percentile in enumerate(feature_percentiles):
            if feature_percentile < percentile_threshold/2 or feature_percentile > 100 - percentile_threshold/2:
                outliers.add(feature_dict[feature][index][0])

# Manuall review the outlier list and add the outlier to remove to the list below. The outlier that
# I found but didn't remove are in the comment below
#'SHAPIRO RICHARD S''LAVORATO JOHN J''DELAINEY DAVID W','BOWEN JR RAYMOND M''BELDEN TIMOTHY N', 
for outlier in ['BELFER ROBERT', 'BHATNAGAR SANJAY', 'KAMINSKI WINCENTY J', 'TOTAL']:
    enron_data_dict.pop(outlier)


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
def parameter_tuning():
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    # Test paramter for decisiontree
    '''
    tuned_parameters = [{
        'criterion': ['gini', 'entropy'], 
        'max_features': ['auto', 'sqrt', 'log2', None], 
        'min_samples_split': [2, 5, 10, 20], 
        'min_samples_leaf': [1, 2, 5, 10]
    }]
    
    # Test paramter for SVC
    tuned_parameters = [{
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid'], 
        'C': [1, 10, 100, 1000, 5000, 10000]
    }]
    '''
    # Test parameter for AdaBoost
    tuned_parameters = [{
        'n_estimators': [10, 20, 50, 70, 100, 200, 300], 
        'learning_rate': [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    }]

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
