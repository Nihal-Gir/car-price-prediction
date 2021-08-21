''' Car Price Prediction '''

import pandas as pd
import seaborn as sns

df=pd.read_csv("D:/Studies/Course Material/car_price_prediction_project/car data.csv")

df.head()

df.dtypes

df.columns

col=[col for col in df.columns if df[col].dtypes=='object']

col.pop(0)

# to find the unique categories for the categorical variables
for i in col:
    print(str(i)+" - " + str(df[i].unique()))

# to find the count of unique categories for the categorical variables
for i in col:
  print(df[i].value_counts())

df['Owner'].value_counts()

df.isnull().sum()

df.describe()

# dropping the car_name feature
df.drop(['Car_Name'],axis=1,inplace=True)

''' We will create a new feature called number of years to find the age of the car '''

df['Current_Year']=2021

df['Age_Car']=df['Current_Year'] - df['Year']

# dropping the year and current_year feature
df.drop(['Year','Current_Year'],axis=1,inplace=True)

dummy=pd.get_dummies(df[col])

df=pd.concat([df,dummy],axis=1)

df.drop(col,axis=1,inplace=True)

df.corr()

sns.pairplot(df)

# creating a heatmap
corrmat = df.corr()
sns.heatmap(corrmat,annot=True,cmap='RdYlGn')

x=df.iloc[:,1:]
y=df['Selling_Price']

# feature importance
from sklearn.ensemble import ExtraTreesRegressor
imp = ExtraTreesRegressor()
imp.fit(x,y)

imp_features=pd.Series(imp.feature_importances_)
imp_features.index=x.columns
imp_features.sort_values(ascending=False)

imp_features.nlargest(15).plot(kind='bar')

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

''' building different models to check which model works the best '''

# RandomForest Regressor
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()

# XGBoost Regressor
import xgboost
xgb=xgboost.XGBRegressor()

# Linear Regression
from sklearn.linear_model import LinearRegression
lr=LinearRegression()

# Ridge Regression
from sklearn.linear_model import Ridge
ridge=Ridge()

# Lasso Regression
from sklearn.linear_model import Lasso
lasso=Lasso()

models=[rf,xgb,lr,ridge,lasso]

def accuracy(i): # function to check the accuracy score
  acc=i.min()/i.max()
  return acc*100

acc_scores=[]

def acc_mod(list_model,acc):
    for i in list_model:
        m=i.fit(x_train,y_train)
        p=pd.Series(m.predict(x_test))
        df=pd.DataFrame(columns=['a','b'])
        df['a']=y_test
        df['b']=p
        acc_scores.append((df.apply(acc,axis=1)).mean())
    
acc_mod(models,accuracy)

accuracy_scores=pd.DataFrame(columns=['Model Name','Accuracy Score'])
accuracy_scores['Model Name']=pd.Series(['RandomForest Regression','XGB Regression','Linear Regression'
                                         ,'Ridge Regression','Lasso Regression'])
accuracy_scores['Accuracy Score']=pd.Series(acc_scores)

'''lasso regression has the highest accuracy score so we'll consider it for further steps'''

''' creating a pickel file of the model '''

import os 
import pickle

os.getcwd()

file = open('lasso_regressor.pkl','wb')
# creating and opening a file named "lasso_regressor.pkl" in "wb(write byte") mode

# dumping our model and it's parameters into the file opened above
pickle.dump(lasso,file)

file.close() # closing the file

