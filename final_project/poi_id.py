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
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_selection import f_classif, SelectKBest, SelectPercentile
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
#features_list = ['poi', 'from_poi_ratio', 'to_poi_ratio', 'exercised_stock_ratio']
#features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'total_stock_value', 'shared_receipt_with_poi', 'long_term_incentive', 'exercised_stock_options', 'from_messages', 'other', 'from_poi_to_this_person', 'from_this_person_to_poi', 'deferred_income', 'expenses', 'restricted_stock', 'director_fees', 'from_poi_ratio', 'to_poi_ratio', 'exercised_stock_ratio']
features_list = ['poi', 'salary', 'total_payments', 'loan_advances', 'bonus', 'total_stock_value', 'long_term_incentive', 'exercised_stock_options', 'restricted_stock', 'to_poi_ratio', 'from_poi_ratio']



### Load the dictionary containing the dataset
enron_data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers
enron_data_dict.pop('TOTAL')

### Task 3: Create new feature(s)
for person, data in enron_data_dict.items():
    #from poi ratio
    if data['to_messages'] != 0 and data['to_messages'] != 'NaN' and data["from_poi_to_this_person"] != 'NaN':
            enron_data_dict[person]['from_poi_ratio'] = float(data['from_poi_to_this_person']) / float(data['to_messages'])
    else:
        enron_data_dict[person]['from_poi_ratio'] = 'NaN'
    #to poi ratio
    if data['from_messages'] != 0 and data['from_messages'] != 'NaN' and data['from_this_person_to_poi'] != 'NaN':
            enron_data_dict[person]['to_poi_ratio'] = float(data['from_this_person_to_poi']) / float(data['from_messages'])
    else:
        enron_data_dict[person]['to_poi_ratio'] = 'NaN'
    #exercised_stock_ratio
    if data['exercised_stock_options'] != 0 and data['exercised_stock_options'] != 'NaN' and data['salary'] != 'NaN':
            enron_data_dict[person]['exercised_stock_ratio'] = float(data['exercised_stock_options']) / float(data['salary'])
            if enron_data_dict[person]['exercised_stock_ratio'] == 1:
                print enron_data_dict[person]
    else:
        enron_data_dict[person]['exercised_stock_ratio'] = 'NaN'
    

### Store to my_dataset for easy export below.
my_dataset = enron_data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, remove_NaN=False, sort_keys = False)
labels, features = targetFeatureSplit(data)

imp = Imputer(missing_values='NaN', strategy='mean', copy=True)
features = imp.fit_transform(features)
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# Create an overfit decision tree and see which element is most powerful
#print features

selector = SelectKBest(f_classif, k=10)
selector.fit(features, labels)
index = selector.get_support()
new_feature_list = []
for i, value in enumerate(index):
    if value == True:
        new_feature_list.append(features_list[i+1])
print new_feature_list


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
clf = AdaBoostClassifier()
clf.fit(features, labels)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)