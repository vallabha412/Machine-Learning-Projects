#Importing required libraries
import timeit
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

#Reading the dataset
df = pd.read_csv('creditcard.csv')
#print(df.shape)

#Separating the columns as X and Y
x = df.iloc[:,1:-1]
y = df.iloc[:,-1]

#Plot before applying Algorithms
col = list(df.columns)
for i in range(1,len(col)-1):
    plt.subplot(5,6,i)
    plt.plot(x[col[i]],y)
    plt.xlabel(col[i])
    plt.ylabel('Class')
    plt.tight_layout()
plt.show()

#Splitting the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

# Applying different algorithms
#KNN
from sklearn.neighbors import KNeighborsClassifier
acc = {}
for i in range(1,50):
     kn = KNeighborsClassifier(n_neighbors=i,metric='euclidean')
     kn.fit(x_train,y_train)
     y_pred = kn.predict(x_test)
     acc[i] = metrics.accuracy_score(y_test,y_pred)
     if i == 49:
          kn = KNeighborsClassifier(n_neighbors=max(acc,key=acc.get),metric='euclidean')

          
#ADA
from sklearn.ensemble import AdaBoostClassifier
ad = AdaBoostClassifier(n_estimators=5,learning_rate=1)

     
#Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = 'entropy', max_depth=3)


#SVM
from sklearn.svm import SVC
s = SVC()



algorithms = [kn,ad,dt,s]
for i in algorithms:
     start = timeit.default_timer()
     i.fit(x_train,y_train)
     pred_y = i.predict(x_test)
     print('Accuracy : ',metrics.accuracy_score(y_test,y_pred))
     print('Confusion Matrix : ',metrics.confusion_matrix(y_test,y_pred))
     stop = timeit.default_timer()
     print('Time: ', stop - start,i)
     print('\n')

#Plot after applying algorithms
for i in range(1,len(col)-1):
    plt.subplot(5,6,i)
    plt.plot(x[col[i]],y)
    plt.xlabel(col[i])
    plt.ylabel('Class')
    plt.tight_layout()
plt.show()

plt.plot(y_test,y_pred)
plt.xlabel('Y_TEST')
plt.ylabel('Y_PRED')
plt.show()
