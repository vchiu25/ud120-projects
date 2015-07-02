#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )

#remove outlier
#sort dicationary
sorted_data_dict = []
sorted_data_dict = sorted(data_dict, key=lambda x: data_dict[x]['bonus'] if data_dict[x]['bonus'] != 'NaN' and data_dict[x]['salary'] > 1000000 else 0, reverse=True)
data_dict.pop('TOTAL')
print sorted_data_dict


features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

### your code below

#loop through dictionary to find max
max_salary_index = data_dict.keys()[0]
max_bonus_index = data_dict.keys()[0]
for person in data_dict.keys():
	if data_dict[person]['salary'] > data_dict[max_salary_index]['salary'] and data_dict[person]['salary'] != 'NaN':
		max_salary_index = person
	if data_dict[person]['bonus'] > data_dict[max_bonus_index]['bonus'] and data_dict[person]['bonus'] != 'NaN':
		max_bonus_index = person

#print data_dict.items()
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()