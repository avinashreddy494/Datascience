# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 14:35:19 2023

@author: 91939
"""



import pandas as pd

messages = pd.read_csv('C:/Users/91939/Desktop/NLP/datasets/spam_ham_dataset.csv', on_bad_lines='skip')

messages=messages.dropna()



import nltk
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.corpus import stopwords
import sklearn
import re

wl=WordNetLemmatizer()
ps=PorterStemmer()
x=messages['text'][:1000]
y=messages['label'][:1000]
x_data=[]
for i in range(len(x)):
    review=re.sub('[^a-zA-Z]', " ", x[i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review= " ".join(review)
    x_data.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x=cv.fit_transform(x_data).toarray()
y=pd.get_dummies(y)
y=y.iloc[:,1].values



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.8,random_state=0)



from sklearn.naive_bayes import MultinomialNB
mb=MultinomialNB()
mb.fit(x_train,y_train)
y_pred=mb.predict(x_test)



from sklearn.metrics import confusion_matrix 
cm=confusion_matrix(y_test, y_pred)


from sklearn.metrics import accuracy_score
accuaracy=accuracy_score(y_test, y_pred)




    