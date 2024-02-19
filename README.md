# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the modules pandas , numpy, matplotlib.pyplot and from sklearn.metrics import mean_absolute_error, mean_squared_error
2. Read the CSV file
3. Create a variable X to store the values of the independent variable
4. Create a variable Y to store the values of the dependent variable
5. Split the training and test data by importing train_test_split from sklearn.model_selection; choose the test size as 1/3 and the random state as 0
6. Use the LinearRegression function from sklearn.linear_model to predict the values
7. Display Y_pred
8. Display Y_test
9. Plot the graph for training data
10. Plot the graph for test data

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Joel John Jobinse
RegisterNumber:  212223240062
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head()

df.tail()

X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

#splitting training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
Y_pred

Y_test

#graph plot for training data
plt.scatter(X_train,Y_train,color='orange')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(X_test,Y_test,color='orange')
plt.plot(X_test,regressor.predict(X_test),color='red')
plt.title("Hours vs Scores(Test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
## Output:
![training_data_graph](https://github.com/joeljohnjobinse/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138955488/cf9d622c-25ee-4eda-aa8a-50a3868bf996)
![test_data_graph](https://github.com/joeljohnjobinse/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138955488/d20ab1d2-7fda-4961-9192-34319a98bb03)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
