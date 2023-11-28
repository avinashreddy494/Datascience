# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 14:23:05 2023

@author: Avinash Reddy Konda
"""

import nltk
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.corpus import stopwords


'''
tf:-


tf:- no.of repating words in a sentence / no.of words in a sentence

idf:-

log(no.of sentences / no.of sentences containg words )

tf*idf

'''
paragraph = """I have three visions for India. In 3000 years of our history, people from all over 
               the world have come and invaded us, captured our lands, conquered our minds. 
               From Alexander onwards, the Greeks, the Turks, the Moguls, the Portuguese, the British,
               the French, the Dutch, all of them came and looted us, took over what was ours. 
               Yet we have not done this to any other nation. We have not conquered anyone. 
               We have not grabbed their land, their culture, 
               their history and tried to enforce our way of life on them. 
               Why? Because we respect the freedom of others.That is why my 
               first vision is that of freedom. I believe that India got its first vision of 
               this in 1857, when we started the War of Independence. It is this freedom that
               we must protect and nurture and build on. If we are not free, no one will respect us.
               My second vision for India’s development. For fifty years we have been a developing nation.
               It is time we see ourselves as a developed nation. We are among the top 5 nations of the world
               in terms of GDP. We have a 10 percent growth rate in most areas. Our poverty levels are falling.
               Our achievements are being globally recognised today. Yet we lack the self-confidence to
               see ourselves as a developed nation, self-reliant and self-assured. Isn’t this incorrect?
               I have a third vision. India must stand up to the world. Because I believe that unless India 
               stands up to the world, no one will respect us. Only strength respects strength. We must be 
               strong not only as a military power but also as an economic power. Both must go hand-in-hand. 
               My good fortune was to have worked with three great minds. Dr. Vikram Sarabhai of the Dept. of 
               space, Professor Satish Dhawan, who succeeded him and Dr. Brahm Prakash, father of nuclear material.
               I was lucky to have worked with all three of them closely and consider this the great opportunity of my life. 
               I see four milestones in my career"""
             
                
sentences=nltk.sent_tokenize(paragraph)
corpus_stem=[]
corpus_lem=[]
ps=PorterStemmer()
wl=WordNetLemmatizer()
for i in range(len(sentences)):
    result=re.sub('[^a-zA-Z]'," ",sentences[i])
    result=result.lower()
    result=result.split()
    result1=[ps.stem(j) for j in result if j not in set(stopwords.words("english"))]
    result2=[wl.lemmatize(j) for j in result if j not in set(stopwords.words("english"))]
    result2=" ".join(result2)
    result1= " ".join(result1)
    corpus_stem.append(result1)
    corpus_lem.append(result2)

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

tdidf=TfidfVectorizer()

x1=tdidf.fit_transform(corpus_lem).toarray()

x2=tdidf.fit_transform(corpus_stem).toarray()
