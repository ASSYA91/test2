import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
data= pd.read_csv('kc_house_data.csv')
print(data.head())
print(data.shape)
print(data.columns)
data.dropna(axis=0,inplace=True)
print(data.head())
print(data.shape)
print(data.isnull().sum().sum())
print(data.describe())
data['price'].hist()
#sns.pairplot(data)
data.drop(['id','date','sqft_lot15','condition','yr_built','zipcode','long','sqft_lot','sqft_living15','yr_renovated'],axis=1,inplace=True)
print(data.columns)
def plot_correlation_map(df):
    corr = df.corr()
    s , ax = plt.subplots( figsize =( 12 , 10 ) )

    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

    s = sns.heatmap(

        corr, 

        cmap = cmap,

        square=True, 

        cbar_kws={ 'shrink' : .9 }, 

        ax=ax, 

        annot = True, 

        annot_kws = { 'fontsize' : 12 }
)
#plot_correlation_map(data)
#sns.pairplot(data)

#we did the covariance figure to see the effect of every feature, than we eliminate 
#the features that doesn't affect the target than we drop the others 

y= data['price']
X= data.drop(['price'],axis=1)
#plt.scatter(data['sqft_living','price'])
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
model = LinearRegression()
model.fit(X_train, y_train)
model.score(X, y)
predictions= model.predict(X_test)
plt.figure(figsize=(12,12))
plt.scatter(y_test, predictions)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color='red')
print("MSE",mean_squared_error(y_test, predictions))
print("R squared",metrics.r2_score(y_test,predictions))

#print(X)
#print(y)
# polynomial regression 
lg= LinearRegression()
poly= PolynomialFeatures(degree=6)
#X_= poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
X_train_fit=poly.fit_transform(X_train)
lg.fit(X_train_fit, y_train)
X_test_= poly.fit_transform(X_test)
predictions_=lg.predict(X_test_)
plt.figure(figsize=(12,12))
plt.scatter(y_test, predictions_)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color='yellow')
plt.xlabel("Prix")
plt.ylabel("Prediction de prix")
plt.title("Prix reels vs predictions methode polynomial")
print("MSE polynomial",mean_squared_error(y_test, predictions_))
print("R squared polynomial",metrics.r2_score(y_test,predictions_))

