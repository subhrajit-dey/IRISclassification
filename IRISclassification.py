# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 16:42:53 2022

@author: SUBHRAJIT_DEY
"""

from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
Y = iris.target

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=1234,stratify=Y)


#Implementation of SVM
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# =============================================================================
# svc = SVC(kernel = 'rbf', gamma = 1.0)
# =============================================================================
#rbf
#‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’

svc = SVC(kernel = 'rbf', gamma = 1.0)

svc.fit(x_train,y_train)

y_predict = svc.predict(x_test)

cm_rbf01 = confusion_matrix(y_test,y_predict)
score = svc.score(x_test,y_test)