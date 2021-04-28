import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn

from pylab import rcParams
from pandas import Series, DataFrame

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

##EXPLORATORY ANALYIS
#Read in dataset
address = HeartDiease = "/Users/jerem/Documents/CPSC_Courses/CPSC_392/heart.csv"
heart_data = pd.read_csv(address)
heart_data.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fps', 'restecg', 'thalach',
'exang', 'oldpeak' , 'slope', 'ca' ,'thal', 'target']
print(heart_data.head())

#Here we can see the average age of a patient is about 55-57
sb.boxplot(x='age', data = heart_data, palette = 'hls')
#plt.show()

#We can see that average age for females is slightly higher then females
#In other words, the average female patient admitted to the hospital is older
#Average female age is about 57, for male it's 54
sb.boxplot(x='sex', y='age', data = heart_data, palette = 'hls')
#plt.show()

#Looking to see a relationship between age and resting blood pressure
#Graph doesn't clearly show that variables are dependent
#Could be a good set for linear regression
#Maybe there is a difference between the genders?
sb.regplot(x='age', y='trestbps', data = heart_data, scatter = True)
#plt.show()

#distribution plot that shows resting blood pressure
#average is about 130 across all patients
restpbs = heart_data['trestbps']
sb.distplot(restpbs)
#plt.show()

#filtered dataset to only males
males_df = heart_data.query('sex>0')
print(males_df)

#filtered dataset to only males
females_df = heart_data.query('sex<1')
print(females_df)

#looking into relationship between age and resting blood pressure of men
#Doesn't seem to be any different correlation compared to before
sb.regplot(x = 'age', y = 'trestbps', data = males_df, scatter = True)
#plt.show()

#looking into relationship between age and resting blood pressure of females
#Looks to be a bit weaker then the correlation between males
sb.regplot(x = 'age', y = 'trestbps', data = females_df, scatter = True)
#plt.show()

##LINEAR REGRESSION
#Going to perform LR to see a correlation in age and resting blood pressure for both men and women
#For Males:
age = males_df['age'].values
restbps = males_df['trestbps'].values
plt.plot(age, restbps, 'r^')
plt.title('Age of Male Patients vs. Resting Blood Pressure')
plt.xlabel('Age of Patient')
plt.ylabel('Resting Blood Pressure (mmHG)')
plt.show()

X = age
Y = restbps

#split data into test and training data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#reshape variables
x_train= x_train.reshape(-1, 1)
y_train= y_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)

#create our linear regression function
LinReg = LinearRegression()

#fit to our model

LinReg.fit(x_train,y_train)
#print the intercept and coefficient
#y = mx+b
#y = 107.82x + 0.422

print(LinReg.intercept_, LinReg.coef_)

#creates a score for our model
#0.04
#model is pretty weak
print(LinReg.score(x_train, y_train))

#creates a list of predictions
y_pred = LinReg.predict(x_test)
print(y_pred)

#For Females:
age = females_df['age'].values
restbps = females_df['trestbps'].values
plt.plot(age, restbps, 'r^')
plt.title('Age of Female Patients vs. Resting Blood Pressure')
plt.xlabel('Age of Patient')
plt.ylabel('Resting Blood Pressure (mmHG)')
plt.show()

X = age
Y = restbps

#split data into test and training data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#reshape variables
x_train= x_train.reshape(-1, 1)
y_train= y_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)

#create our linear regression function
LinReg = LinearRegression()

#fit to our model

LinReg.fit(x_train,y_train)
#print the intercept and coefficient
#y = mx+b
#y = 89.89 + 0.8x

print(LinReg.intercept_, LinReg.coef_)

#creates a score for our model
#0.14
#model is stronger
print(LinReg.score(x_train, y_train))

#creates a list of predictions
y_pred = LinReg.predict(x_test)
print(y_pred)
