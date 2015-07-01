#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC
#smaller dataset
features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]
#training and testing
for c_parameter in [1]:
    clf = SVC(kernel="rbf", C = c_parameter)
    t0 = time()
    clf.fit(features_train, labels_train)
    print "training time: ", round(time()-t0,2), "s"
    t0 = time()
    print clf.score(features_test, labels_test)
    print "testing time: ", round(time()-t0,2), "s"
#print result
prediction = clf.predict(features_test)
for i in [10,26,50]:
    print prediction[i]
print sum(prediction)
#########################################################


