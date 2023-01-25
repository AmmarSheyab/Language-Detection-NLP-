# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 14:38:06 2022

@author: ASUS
"""

import numpy as np
import pandas as pd

df = pd.read_csv('Language Detection.csv',usecols=(['Text','Language']))
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0,len(df)):
    
  new = re.sub('[^a-zA-Z]', ' ', df['Text'][i])
  new=new.lower()
  new=new.split()
  ps=PorterStemmer()
  new=[ps.stem(word) for word in new if not word  in set(stopwords.words('english'))] 
  new=' '.join(new)
  corpus.append(new)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5150)
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)



from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#print('naive bayes \n',cm)



from sklearn.model_selection import train_test_split
X_tree_train, X_tree_test, y_tree_train, y_tree_test = train_test_split(X, y, test_size = 0.20, random_state = 0)



from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_tree_train,y_tree_train)
y_pred2=classifier.predict(X_tree_test)
cm1=confusion_matrix(y_tree_test,y_pred2)
#print('decision tree \n',cm1)




from sklearn.model_selection import train_test_split
X_forst_train, X_forst_test, y_forst_train, y_forst_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=50,criterion='entropy',random_state=0)
classifier.fit(X_forst_train,y_forst_train)
y_pred3=classifier.predict(X_forst_test)
cm2=confusion_matrix(y_forst_test,y_pred3)
#print('random forset\n',cm2)





