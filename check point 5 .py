import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression

titanic = sns.load_dataset('titanic')
print(titanic.shape)
print(titanic.head())
titanic= titanic[['survived','pclass','sex','age']]
titanic.dropna(axis=0, inplace=True)
titanic['sex'].replace(['male','female'],[0,1],inplace=True)
print(titanic.head())
Model= LogisticRegression()
y= titanic['survived']
X= titanic.drop('survived', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
Model.fit(X_train, y_train)

y_pred = Model.predict(X_test)
print("accuracy={:.2f}".format(Model.score(X_test, y_test)))
def survie(Model,pclass=3,sex=1,age=31):
    x= np.array([pclass,sex,age]).reshape(1,3)
    print(Model.predict(x))
survie(Model)
#sns.regplot(x='age',y='survived',data=titanic)
confusion_matrix=pd.crosstab(y_test, y_pred,rownames=['actual'],colnames=['predicted'])
sns.heatmap(confusion_matrix,annot=True)
