import pandas as pd
import numpy as np
from sklearn import metrics

df = pd.read_csv('creditcard.csv')
print(df.shape)
x = df.iloc[:,0:-1]
y = df.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)
'''
from sklearn.ensemble import AdaBoostClassifier
ad = AdaBoostClassifier(n_estimators=5,learning_rate=1)
ad.fit(x_train,y_train)
y_pred1 = ad.predict(x_test)
print('Accuracy :',metrics.accuracy_score(y_test,y_pred1))
print('Confusion_matrix : ',metrics.confusion_matrix(y_test,y_pred1))

acc1 = {}
for i in range(1,50):
     ad = AdaBoostClassifier(n_estimators=i,learning_rate=1)
     ad.fit(x_train,y_train)
     y_pred1 = ad.predict(x_test)
     acc1[i] = metrics.accuracy_score(y_test,y_pred1)
     if i == 49:
          ad = AdaBoostClassifier(n_estimators=max(acc1,key=acc.get),metric='euclidean')
          ad.fit(x_train,y_train)
          y_pred1 = kn.predict(x_test)
          print(max(acc1,key=acc1.get))
          print('Accuracy :',metrics.accuracy_score(y_test,y_pred1))
          print('Confusion_matrix : ',metrics.confusion_matrix(y_test,y_pred1))


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
dt.fit(x_train,y_train)
y_pred2 = dt.predict(x_test)
print('Accuracy :',metrics.accuracy_score(y_test,y_pred2))
print('Confusion_matrix : ',metrics.confusion_matrix(y_test,y_pred2))
'''

from sklearn.svm import SVC
s = SVC()
s.fit(x_train,y_train)
y_pred3 = s.predict(x_test)
print('Accuracy :',metrics.accuracy_score(y_test,y_pred3))
print('Confusion_matrix : ',metrics.confusion_matrix(y_test,y_pred3))
