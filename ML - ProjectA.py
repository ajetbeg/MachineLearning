#!/usr/bin/env python
# coding: utf-8

# In[89]:


#Multiple linear regression with 2 different variables
#First we take 5 years of historical dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv("C:\\Users\\ajetb\\Desktop\\RS5.csv")

# Dependent variable Y
close_prices = data['Close']

#Open + Volume = independent variables X
data2 = np.genfromtxt("C:\\Users\\ajetb\\Desktop\\RS5.csv", dtype='float', delimiter=',', skip_header=1, usecols=(1,6))
X = np.array(data2)
print(X)

# Split the data into training/testing sets
X_train = np.array(X[:1007]).reshape((-1, 2)) #1007 is the 80% of elements
X_test = np.array(X[1007:]).reshape((-1, 2)) 

# Split the targets into training/testing sets
Y_train = close_prices[:1007] 
Y_test = close_prices[1007:] 


regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, Y_train)

# Make predictions using the testing set
Y_pred = regr.predict(X_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(Y_test, Y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(Y_test, Y_pred))


# In[90]:


#Multiple linear regression with 7 years of historical dataset

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv("C:\\Users\\ajetb\\Desktop\\RS7.csv")

# Dependent variable Y
close_prices = data['Close']

#Open + Volume = independent variables X
data2 = np.genfromtxt("C:\\Users\\ajetb\\Desktop\\RS7.csv", dtype='float', delimiter=',', skip_header=1, usecols=(1,6))
X = np.array(data2)
print(X)

# Split the data into training/testing sets
X_train = np.array(X[:1410]).reshape((-1, 2)) #1007 is the 80% of elements
X_test = np.array(X[1410:]).reshape((-1, 2)) 

# Split the targets into training/testing sets
Y_train = close_prices[:1410] 
Y_test = close_prices[1410:] 


regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, Y_train)

# Make predictions using the testing set
Y_pred = regr.predict(X_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(Y_test, Y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(Y_test, Y_pred))


# In[91]:


#Multiple linear regression with 10 years of historical dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv("C:\\Users\\ajetb\\Desktop\\RS10.csv")

# Dependent variable Y
close_prices = data['Close']

#Open + Volume = independent variables X
data2 = np.genfromtxt("C:\\Users\\ajetb\\Desktop\\RS10.csv", dtype='float', delimiter=',', skip_header=1, usecols=(1,6))
X = np.array(data2)
print(X)

# Split the data into training/testing sets
X_train = np.array(X[:2013]).reshape((-1, 2)) #1007 is the 80% of elements
X_test = np.array(X[2013:]).reshape((-1, 2)) 

# Split the targets into training/testing sets
Y_train = close_prices[:2013] 
Y_test = close_prices[2013:] 


regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, Y_train)

# Make predictions using the testing set
Y_pred = regr.predict(X_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(Y_test, Y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(Y_test, Y_pred))


# In[93]:


#Multiple linear regression with 4 different variables 
#First we take 5 years of historical dataset

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv("C:\\Users\\ajetb\\Desktop\\RS5.csv")

# Dependent variable Y
close_prices = data['Close']

#Open + Volume = independent variables X
data2 = np.genfromtxt("C:\\Users\\ajetb\\Desktop\\RS5.csv", dtype='float', delimiter=',', skip_header=1, usecols=(1,2,3,6))
X = np.array(data2)
print(X)

# Split the data into training/testing sets
X_train = np.array(X[:1007]).reshape((-1, 4)) 
X_test = np.array(X[1007:]).reshape((-1, 4)) 
# Split the targets into training/testing sets
Y_train = close_prices[:1007] 
Y_test = close_prices[1007:] 


regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, Y_train)

# Make predictions using the testing set
Y_pred = regr.predict(X_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(Y_test, Y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(Y_test, Y_pred))


# In[94]:



#Multiple linear regression with 4 different variables 
#7 years of historical dataset

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv("C:\\Users\\ajetb\\Desktop\\RS7.csv")

# Dependent variable Y
close_prices = data['Close']

#Open + Volume = independent variables X
data2 = np.genfromtxt("C:\\Users\\ajetb\\Desktop\\RS7.csv", dtype='float', delimiter=',', skip_header=1, usecols=(1,2,3,6))
X = np.array(data2)
print(X)

# Split the data into training/testing sets
X_train = np.array(X[:1410]).reshape((-1, 4)) 
X_test = np.array(X[1410:]).reshape((-1, 4)) 
# Split the targets into training/testing sets
Y_train = close_prices[:1410] 
Y_test = close_prices[1410:] 


regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, Y_train)

# Make predictions using the testing set
Y_pred = regr.predict(X_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(Y_test, Y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(Y_test, Y_pred))


# In[95]:


#Multiple linear regression with 4 different variables 
#10 years of historical dataset of  Reliance Steel & Aluminum Co. (RS) 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv("C:\\Users\\ajetb\\Desktop\\RS10.csv")

# Dependent variable Y
close_prices = data['Close']

#Open + Volume = independent variables X
data2 = np.genfromtxt("C:\\Users\\ajetb\\Desktop\\RS10.csv", dtype='float', delimiter=',', skip_header=1, usecols=(1,2,3,6))
X = np.array(data2)
print(X)

# Split the data into training/testing sets
X_train = np.array(X[:2013]).reshape((-1, 4)) 
X_test = np.array(X[2013:]).reshape((-1, 4)) 
# Split the targets into training/testing sets
Y_train = close_prices[:2013] 
Y_test = close_prices[2013:] 


regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, Y_train)

# Make predictions using the testing set
Y_pred = regr.predict(X_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(Y_test, Y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(Y_test, Y_pred))


# In[68]:


#Graphical Representation of Multiple Linear Regression (with 2 independent variables x1 and x2)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
# Load the dataset
data = pd.read_csv("C:\\Users\\ajetb\\Desktop\\RS10.csv")
# Dependent variable Y
Y = data['Close']
print(Y)
# Independent variables X1 and X2
X = data[['Open', 'Volume']].to_numpy()
X1 = data['Open']
print(X1)
X2 = data['Volume']
print(X2)
# create training and test splits
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)
X_train2 = np.array(X_train).reshape((-1, 2)) 
X_test2 = np.array(X_test).reshape((-1, 2))
X1_test = X_test2[:, 0]
X2_test = X_test2[:, 1]
x1_pred = np.linspace(80, 1300, 30)   # range of open values np.linspace(min, max, datapoints)
x2_pred = np.linspace(10000, 3000000, 30)  # range of volume values
xx1_pred, xx2_pred = np.meshgrid(x1_pred, x2_pred)
# model_viz = np.array([xx1_pred.flatten(), xx2_pred.flatten()]).T
# Create a model and fit it
ols = linear_model.LinearRegression()
model = ols.fit(X_train2, Y_train)
# Use the model for making Predictions
Y_pred = model.predict(X_test2)
print('predicted response:', Y_pred, sep='\n')
# Get results
r2 = model.score(X_test2, Y_pred)
print('coefficient of determination:', r2)
print('intercept:', model.intercept_)
print('coefficients:', model.coef_)
plt.style.use('default')
fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')
axes = [ax1, ax2, ax3]
for ax in axes:
    ax.plot(X1_test, X2_test, Y_test, color='k', zorder=15, linestyle='none', 
marker='o', alpha=0.5)
    ax.scatter(X1_test, X2_test, Y_pred, facecolor=(0,0,0,0), s=20, 
edgecolor='#70b3f0')
    ax.set_xlabel('X1', fontsize=12)
    ax.set_ylabel('X2', fontsize=12)
    ax.set_zlabel('Y', fontsize=12)
    ax.locator_params(nbins=4, axis='x')
    ax.locator_params(nbins=5, axis='x')
ax1.view_init(elev=28, azim=120)
ax2.view_init(elev=4, azim=114)
ax3.view_init(elev=60, azim=165)
fig.suptitle('$R^2 = %.2f$' % r2, fontsize=20)
fig.tight_layout()


# In[76]:


#Graphical Representation of Multiple Linear Regression (with 2 independent variables x1 and x2)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
# Load the dataset
data = pd.read_csv("C:\\Users\\ajetb\\Desktop\\RS10.csv")
# Dependent variable Y
Y = data['Close']
print(Y)
# Independent variables X1 and X2
X = data[['Open', 'Volume']].to_numpy()
X1 = data['Open']
print(X1)
X2 = data['Volume']
print(X2)
# create training and test splits
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)
X_train2 = np.array(X_train).reshape((-1, 2)) 
X_test2 = np.array(X_test).reshape((-1, 2))
X1_test = X_test2[:, 0]
X2_test = X_test2[:, 1]
x1_pred = np.linspace(50, 210, 30)   # range of open values np.linspace(min, max, datapoints) of my dataset
x2_pred = np.linspace(67200, 2719900, 30)  # range of volume values of my dataset
xx1_pred, xx2_pred = np.meshgrid(x1_pred, x2_pred)
# model_viz = np.array([xx1_pred.flatten(), xx2_pred.flatten()]).T
# Create a model and fit it
ols = linear_model.LinearRegression()
model = ols.fit(X_train2, Y_train)
# Use the model for making Predictions
Y_pred = model.predict(X_test2)
print('predicted response:', Y_pred, sep='\n')
# Get results
r2 = model.score(X_test2, Y_pred)
print('coefficient of determination:', r2)
print('intercept:', model.intercept_)
print('coefficients:', model.coef_)
plt.style.use('default')
fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')
axes = [ax1, ax2, ax3]
for ax in axes:
    ax.plot(X1_test, X2_test, Y_test, color='k', zorder=15, linestyle='none', 
marker='o', alpha=0.5)
    ax.scatter(X1_test, X2_test, Y_pred, facecolor=(0,0,0,0), s=20, 
edgecolor='#70b3f0')
    ax.set_xlabel('X1', fontsize=12)
    ax.set_ylabel('X2', fontsize=12)
    ax.set_zlabel('Y', fontsize=12)
    ax.locator_params(nbins=4, axis='x')
    ax.locator_params(nbins=5, axis='x')
ax1.view_init(elev=28, azim=120)
ax2.view_init(elev=4, azim=114)
ax3.view_init(elev=60, azim=165)
fig.suptitle('$R^2 = %.2f$' % r2, fontsize=20)
fig.tight_layout()


# In[ ]:





# In[78]:


#Logistic Regression for predicting if a stock (Close price) goes UP/DOWN 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


import pandas as pd
data=pd.read_csv("C:\\Users\\ajetb\\Desktop\\RS10years.csv")
print(data)
# Use only one feature
X = data['Open']
Y = data['Y_variable']
print(X)
print(Y)
# Split the data into training/testing sets
X_train = np.array(X[:2013]).reshape((-1, 1))
X_test = np.array(X[2013:]).reshape((-1, 1))
# Split the targets into training/testing sets
Y_train = Y[:2013]
Y_test = Y[2013:]
# Create Logistic Regression object and Train the model using the training sets
model = LogisticRegression(solver='liblinear', random_state=0).fit(X_train, Y_train)
# Attributes of the model
print('model.classes_:', model.classes_)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
# Make predictions using the testing set
Y_pred = model.predict_proba(X_test)
print(Y_pred)
print("-------------------------------")
Y_pred2 = model.predict(X_test)
print(Y_pred2)
Y_pred3 = np.array(Y_pred2).reshape(-1, 1)
print("-------------------------------")
accuracy = model.score(Y_pred3, Y_test)
print("accuracy = ", accuracy)
print("-------------------------------")
# print(Y_pred3)
Y_test2 = np.array(Y_test).reshape(-1, 1)
# print(Y_test2)
combined = np.hstack((Y_pred3, Y_test2))
print(combined)
print("-------------------------------")
# Evaluating the Algorithm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(Y_test,Y_pred3))
print(classification_report(Y_test,Y_pred3))
print(accuracy_score(Y_test, Y_pred3))


# In[87]:


#Logistic Regression for predicting if a stock (Close price) goes UP/DOWN with 4 independent variables

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression 
from scipy import stats
import seaborn as sns
# Load the dataset
dataset = pd.read_csv("C:\\Users\\ajetb\\Desktop\\RS10years.csv")
print(dataset)
# split our dataset into its attributes and labels
# we specify features (inputs / X variables)
#   --->     X = dataset[['Open', 'High', 'Low']].to_numpy()
X = pd.DataFrame(dataset, columns=['Open', 'High', 'Low'])
#X.to_csv("C:\\Users\\ajetb\\Desktop\\RS10years.csv", encoding='utf-8', index=False)
# labels (outputs / Y variable) for the model.
#   --->      Y = np.where(dataset['Class'] == 'down', 0, 1)
Y = pd.Series(dataset['Y_variable'])
# create training and test splits
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)
# Y_train2 = np.where(Y_train == 'down', 0, 1)
# Y_test2 = np.where(Y_test == 'down', 0, 1)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
print(X_test)
print("-------------")
print(Y_test)
Y_test.to_csv("C:\\Users\\ajetb\\Desktop\\RS10years.csv", 
encoding='utf-8', index=True)
print(Y_test2)
# Before making any actual predictions, it is always a good practice to scale 
# the features  so that all of them can be uniformly evaluated
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# Create the model 
logreg =  LogisticRegression(solver='liblinear') 
# fit the model with data 
logreg.fit(X_train,Y_train)
#--------------------------------------------
params = np.append(logreg.intercept_,logreg.coef_)
#params = np.append(regr.intercept_,regr.coef_)
# predictions = regr.predict(X_train)
predictions = logreg.predict(X_train)
newX = np.append(np.ones((len(X_train),1)), X_train, axis=1)
MSE = (sum((Y_train-predictions)**2))/(len(newX)-len(newX[0]))
var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
sd_b = np.sqrt(var_b)
ts_b = params/ sd_b
p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-len(newX[0])))) for i in ts_b]
sd_b = np.round(sd_b,3)
ts_b = np.round(ts_b,3)
p_values = np.round(p_values,3)
params = np.round(params,4)
myDF3 = pd.DataFrame()
myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["Probabilities"] = [params,sd_b,ts_b,p_values]
print(myDF3)
#--------------------------------------------
# Make predictions using the testing set
# Y_pred = regr.predict(X_test)
Y_pred=logreg.predict(X_test) 
Y_pred2 = np.rint(Y_pred)
#replace all elements greater than 1 with a new value of 1
Y_pred2[Y_pred2 > 1] = 1
df = pd.DataFrame({'Real Values':Y_test, 'Predicted Values':Y_pred2})
df.to_csv("C:\\Users\\ajetb\\Desktop\\RS10years.csv", encoding='utf-8', 
index=False)
df
# The coefficients
# The intercept of the regression curve
print("Coefficient of intercept: \n", logreg.intercept_)
# The slope  of the regression curve
print("Coefficients of the slope: \n", logreg.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(Y_test, Y_pred2))
mae = mean_absolute_error(Y_test, Y_pred2)
mse = mean_squared_error(Y_test, Y_pred2)
rmse = np.sqrt(mse)
print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')
# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(Y_test, Y_pred2)
print(cnf_matrix)
# Visualizing Confusion Matrix using Heatmap
# Define the two classes 
class_names=[0,1] 
fig, ax = plt.subplots() 
tick_marks = np.arange(len(class_names)) 
plt.xticks(tick_marks, class_names) 
plt.yticks(tick_marks, class_names) 
# create heatmap 
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g') 
ax.xaxis.set_label_position("top") 
plt.tight_layout() 
plt.title('Confusion matrix', y=1.1) 
plt.ylabel('Actual label') 
plt.xlabel('Predicted label')
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred2))
print("Precision:",metrics.precision_score(Y_test, Y_pred2, average='weighted'))
print("Recall:",metrics.recall_score(Y_test, Y_pred2, average='weighted'))


# In[ ]:




