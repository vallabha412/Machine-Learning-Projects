#Importing Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

#Reading the Dataset
df = pd.read_csv("diabetes.csv")

#Separating the Dataset into Dependent and Independent
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

#Scaling the Data
ss = StandardScaler()
x = ss.fit_transform(x)

#Splitting the data into Training Set and Test Set
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=5)

#Implementing Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=3)

#Fitting the data
dt.fit(x_train,y_train)

#Predicting the Targets
y_pred = dt.predict(x_test)

#Calculating the metrics before Tuning
acc = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test,y_pred)
print("------------------Before Tuning-----------------------")
print("Accuracy Score = {} \n\nConfusion Matrix = {}".format(acc,cm))
'''
#Hyperparameter Tuning
param_dict = {
    'criterion':['gini','entropy'],
    'max_depth':range(1,20),
    'min_samples_split':range(1,20),
    'min_samples_leaf':range(1,10)
}
grid = GridSearchCV(DecisionTreeClassifier(),
                    param_grid=param_dict,
                    cv=10,
                    verbose=1,
                    n_jobs=-1)

grid.fit(x_train,y_train)
#print(grid.best_params_)
'''
#Implementing the Decision Tree Classifier with Best Parameters
dt = DecisionTreeClassifier(max_depth=5,min_samples_leaf=3,min_samples_split=3,random_state=29)

#Fitting the Data
dt.fit(x_train,y_train)

#Predicting the Targets
y_pred = dt.predict(x_test)

#Printing the Metrics after Tuning
print("\n\n")
print("-------------------------After Tuning-----------------------")
acc = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test,y_pred)
print("Accuracy Score={}\n\nConfusion Matrix={}".format(acc,cm))

