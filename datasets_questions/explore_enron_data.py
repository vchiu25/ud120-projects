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

#load files
import pickle
enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

#get familiar with data
print len(enron_data)
print enron_data.keys()
print len(enron_data.items()[0][1])
print enron_data.items()[0][1]
count = 0
for i in enron_data:
	if enron_data[i]['poi'] == True:
		print i
		count += 1
print count

print enron_data['PRENTICE JAMES']['total_stock_value']

print enron_data['COLWELL WESLEY']['from_this_person_to_poi']

print enron_data['SKILLING JEFFREY K']['exercised_stock_options']

print enron_data['SKILLING JEFFREY K']['total_payments']
print enron_data['LAY KENNETH L']['total_payments']
print enron_data['FASTOW ANDREW S']['total_payments']

salary_count = 0
email_count = 0
for i in enron_data:
	if enron_data[i]['salary'] != 'NaN':
		salary_count += 1
	if enron_data[i]['email_address'] != 'NaN':
		email_count += 1

print salary_count
print email_count

#play with helper function
import sys, os
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
result = featureFormat(enron_data,["poi", "salary", "bonus"],remove_NaN=False)
print len(result)
targets, features = targetFeatureSplit(result)
print targets
print features

