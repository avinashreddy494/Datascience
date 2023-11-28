# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 16:04:27 2023

@author: Avinash reddy Konda

"""


import pandas as pd

df=pd.read_csv('C:/Users/91939/Desktop/NLP/datasets/Stock_Data.csv',encoding = "ISO-8859-1")

df.head()

print(df['Date'])

train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']

# Removing punctuations
data=train.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ",regex=True, inplace=True)
list1=[ i for i in range(25)]
new_index=[str(i) for i in list1]
data.columns=new_index
for i in new_index:
    data[i]=data[i].str.lower()
headlines = []
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))



test_data=test.iloc[:,2:27]
test_data.replace("[^a-zA-Z]"," ",regex=True, inplace=True)
list1=[ i for i in range(25)]
new_index=[str(i) for i in list1]
test_data.columns=new_index
for i in new_index:
    data[i]=data[i].str.lower()
test_headlines = []
for row in range(0,len(test_data.index)):
    test_headlines.append(' '.join(str(x) for x in test_data.iloc[row,0:25]))

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(ngram_range=(2,2))
x_train=cv.fit_transform(headlines)
x_test=cv.transform(test_headlines)

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=200,criterion='entropy')
rf.fit(x_train,train['Label'])



predictions=rf.predict(x_test)

from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

matrix=confusion_matrix(test['Label'],predictions)

score=accuracy_score(test['Label'],predictions)

report=classification_report(test['Label'],predictions)

