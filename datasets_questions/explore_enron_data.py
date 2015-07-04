#!/usr/bin/python

""" 
    starter code for exploring the Enron dataset (emails + finances) 
    loads up the dataset (pickled dict of dicts)

    the dataset has the form
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person
    you should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import RandomizedPCA
import pickle
import operator
import sys, os
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

#load files
enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

#remove total
enron_data.pop('TOTAL')


#get familiar with data
print 'number of people', len(enron_data)
print 'number of feature', len(enron_data.items()[0][1])

#find the count for each features
feature_set = {}
for feature in enron_data.items()[0][1]:
    feature_set[feature] = 0
for person, data in enron_data.items():
    for i,  feature in enumerate(data):
        if enron_data[person][feature] != 'NaN':
            feature_set[feature] += 1
print sorted(feature_set.items(), key=operator.itemgetter(1), reverse=True)

#look at some specific
'''
print enron_data.keys()
print enron_data.items()[0][1]
print enron_data['PRENTICE JAMES']['total_stock_value']
print enron_data['COLWELL WESLEY']['from_this_person_to_poi']
print enron_data['SKILLING JEFFREY K']['exercised_stock_options']
print enron_data['SKILLING JEFFREY K']['total_payments']
print enron_data['LAY KENNETH L']['total_payments']
print enron_data['FASTOW ANDREW S']['total_payments']
'''

print enron_data['GLISAN JR BEN F']['from_messages']
print enron_data['GLISAN JR BEN F']['to_messages']
print enron_data['GLISAN JR BEN F']['from_poi_to_this_person']
print enron_data['GLISAN JR BEN F']['from_this_person_to_poi']

salary_count = 0
email_count = 0
for i in enron_data:
	if enron_data[i]['salary'] != 'NaN':
		salary_count += 1
	if enron_data[i]['email_address'] != 'NaN':
		email_count += 1

print 'salary_count', salary_count
print 'email_count', email_count

#how many poi has NaN
NaNcount = 0
POIcount = 0
for person, values in enron_data.items():
    if values['poi'] == True:
        POIcount += 1
        if values['total_payments'] == 'NaN':
            NaNcount += 1
print 'NaNcount, POIcount', NaNcount, POIcount



#play with helper function
result = featureFormat(enron_data,["poi", "salary", "exercised_stock_options",  "from_this_person_to_poi", "from_poi_to_this_person"],remove_NaN=False)
targets, features = targetFeatureSplit(result)

#impute missing NaN
imp = Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=True)
imputed_features = imp.fit_transform(features)

#min_max_scale
scaler = MinMaxScaler()
imputed_features = scaler.fit_transform(imputed_features)

# reduce dimension to 2 and plot
n_components = 2
pca = RandomizedPCA(n_components=n_components, whiten=True)
imputed_features_pca = pca.fit_transform(imputed_features)
colors = ['b', 'r']
print range(len(targets))
for ii in range(len(targets)):
    #print ii
    #print type(imputed_features_pca[ii][0])
    plt.scatter(imputed_features_pca[ii][0], imputed_features_pca[ii][1], color = colors[int(targets[ii].item())])

plt.show()



#helper function to see if there are any important feature
def Draw_features(feature_list):
    result = featureFormat(enron_data,feature_list,remove_NaN=False)
    targets, features = targetFeatureSplit(result)
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=True)
    features = imp.fit_transform(features)
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    for ii in range(len(targets)):
        plt.scatter(imputed_features_pca[ii][0], imputed_features_pca[ii][1])
    for ii in range(len(targets)):
        if targets[ii] == 1:
            plt.scatter(imputed_features_pca[ii][0], imputed_features_pca[ii][1], color='r')
    plt.show()


Draw_features(['from_this_person_to_poi', 'from_poi_to_this_person'])

def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than 4 clusters
    colors = ["b", "r", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()

