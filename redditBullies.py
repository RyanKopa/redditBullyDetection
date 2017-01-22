
# coding: utf-8

# In[1]:

#Algorithm that uses NLP and ML to determine if a comment made on social media (specifically reddit)
#Data was gathered from Kaggle's Reddit May 2015 hosted data off of a sql server
#Required packages: numpy, pandas, sklearn
import pandas as pd
import numpy as np

controversiality = pd.read_csv('controversiality.csv', error_bad_lines = False)
noncontroversiality = pd.read_csv('non-controversiality.csv', error_bad_lines = False)
#combine the controversial and noncontroversial data
data = pd.concat((controversiality, noncontroversiality), axis=0, ignore_index=True)


# In[2]:

simple = pd.concat([data['controversiality'], data['body']], axis=1, keys=['author', 'body'])
#drop the null data
simple[pd.isnull(simple).any(axis=1)]
simple = simple.drop(simple.index[[7572,97008]])
simple.info()


# In[3]:

#eliminate non-alpha characters
simple['body'].replace(regex=True,inplace=True,to_replace=r'([^\s\w]|_)+',value=r'')
simple['body'].replace(regex=True,inplace=True,to_replace=r'/s|\n',value=r'')
print(simple.head(10))


# In[4]:

#Create test and training set
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(simple.body, 
                                                                                                 simple.author, 
                                                                                                 test_size=0.2, 
                                                                                                 random_state=42)


# In[5]:

#create vectorizer for feature selection
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_df = 0.1,
                               stop_words='english')


# In[6]:

#transform the reddit comments into tuples of words and their frequency of occurence
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


# In[7]:

#use only the top 100 features
features_train = features_train[:100].toarray()
labels_train   = labels_train[:100]


# In[8]:

#Differente machine learning models used
from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from sklearn.ensemble import RandomForestClassifier
# from xgboost.sklearn import XGBClassifier


# In[10]:

#Run model, get accuracy
clf = DecisionTreeClassifier()
# clf = LogisticRegression()
# clf = RandomForestClassifier()
# clf = xgb.fit(features_train, labels_train)
# clf = XGBClassifier(max_depth=6,
#                     learning_rate=0.1,
#                     )
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print("Accuracy:", accuracy_score(labels_test, pred)) #Beat random guessing!!!


# In[11]:

#rank most important words that determine if a comment is controversial or not
importances = clf.feature_importances_
import numpy as np
indices = np.argsort(importances)[::-1]
print('Feature Ranking: ')
for i in range(10):
    print("{} feature no.{} ({}) {}".format(i+1,indices[i],
                                            importances[indices[i]], 
                                            vectorizer.get_feature_names()[indices[i]]))

