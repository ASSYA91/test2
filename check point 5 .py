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
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics 


titanic = sns.load_dataset('titanic')
print(titanic.shape)
print(titanic.head())
titanic= titanic[['survived','pclass','sex','age']]
titanic['age'] = titanic['age'].fillna(titanic['age'].mode()[0])
print(titanic.shape)
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

# check point #7  


# decision tree 


dataset = sns.load_dataset('titanic')
def preprocessing(new_data):
    new_data['age'].fillna(new_data['age'].mean(),inplace=True)
    new_data['sex'].replace(['male','female'],[0,1],inplace=True)
    

    le = LabelEncoder()
    new_data['class'] = le.fit_transform(new_data['class'].astype(str))
    new_data['who'] = le.fit_transform(new_data['who'].astype(str))


    new_data['survived'].replace(['no','yes'],[0,1],inplace=True)
    print(new_data.head())
    return(new_data)
data= preprocessing(dataset)
#X = pd.get_dummies(X, columns=['cabin'], prefix='cabin')
print(data.columns)



X= data.drop(['survived','embark_town','deck','embarked','alive','alone'],axis=1)
y=data['survived']
print(X.dtypes)
print(data['who'])

#splitting data 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)
#applying a tree algorithm 
tree= tree.DecisionTreeClassifier()
tree.fit(X_train,y_train)
y_pred= tree.predict(X_test)

print("score:{}".format(accuracy_score(y_test, y_pred)))


# applying the random forest algorithm 

clf =RandomForestClassifier(n_estimators=100)

clf.fit(X_train,y_train)

y_pred= clf.predict(X_test)
print("accuracy of random forrest", metrics.accuracy_score(y_test, y_pred))


# le score pour la decision tree est 0.75 pour le random forrest est 0.82



plt.figure(figsize=(30,20))
plot_tree(tree, filled=True, fontsize=11);
plt.show()

'''
import graphviz
dot_data = export_graphviz(tree,out_file=None)

graph=graphviz.Source(dot_data)
graph.render('data')
graph

dtree= tree.DecisionTreeClassifier(criterion="gini",splitter='random',max_leaf_nodes=10,
                                   min_samples_leaf=5,max_depth=5)

'''

