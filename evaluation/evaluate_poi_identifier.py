#!/usr/bin/python


"""
    starter code for the evaluation mini-project
    start by copying your trained/tested POI identifier from
    that you built in the validation mini-project

    the second step toward building your POI identifier!

    start by loading/formatting the data

"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=.3, random_state=42) 
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
print clf.score(features_test, labels_test)
print sum(abs(clf.predict(features_test) - labels_test))

from sklearn.metrics import precision_score, recall_score
print precision_score(clf.predict(features_test), labels_test)
print recall_score(clf.predict(features_test), labels_test)