#!/usr/bin/python


import matplotlib.pyplot as plt
import pickle
import sys
from sklearn.cross_validation import train_test_split, KFold, StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import Imputer, MinMaxScaler
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

# Load data
enron_data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

# Remove total
enron_data_dict.pop('TOTAL')

# Create new feature
# from_poi_ratio: ratio of email form poi to this person
# to_poi_ratio: ratio of email this person to poi
# exercised_stock_ratio: ratio of exercised stock option to total stock value

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
    

# Define list of feature to extract from dictionary  
features_list = ['poi', 'salary', 'exercised_stock_options', 'from_poi_ratio', 'to_poi_ratio']
#"salary", "exercised_stock_options",
#'to_poi_ratio', 'from_poi_ratio'
# Extract list of feature using featureFormat function from tools and the list # of feature defined above.  
data = featureFormat(enron_data_dict, features_list, remove_NaN=True)
labels, features = targetFeatureSplit(data)

imp = Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=True)
imputed_features = imp.fit_transform(features)
'''
scaler = MinMaxScaler()
imputed_features = scaler.fit_transform(imputed_features)
'''
# Cross-Validation split
features_train, features_test, labels_train, labels_test = train_test_split(imputed_features, labels, test_size=.2, random_state=42) 

# Classifer
clf = AdaBoostClassifier()
clf.fit(features_train, labels_train)
print 'precision_score', precision_score(clf.predict(features_test), labels_test)
print 'recall_score', recall_score(clf.predict(features_test), labels_test)

for ii in range(len(labels)):
    if labels[ii] == 1:
        plt.scatter(imputed_features[ii][0], imputed_features[ii][1], color='r', marker='*')
    else:
        plt.scatter(imputed_features[ii][0], imputed_features[ii][1], color='b')
plt.show()
