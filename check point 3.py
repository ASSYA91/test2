import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
#import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns 

data = pd.read_excel('titanic-passengers-1.xlsx')
print(data.head())
print(data.shape)
print(data['Age'])
print(data.columns)
data= data.drop(['SibSp','Parch','Ticket','Fare','Cabin'],axis=1)
print(data.head())
print(data.describe())
print(data.isnull().sum())
print(data.isnull().sum().sum())
data= data.dropna(axis=0,how='any',thresh=None , inplace=False)
print(data.describe())
encoder= LabelEncoder()
data['Survived']= encoder.fit_transform(data['Survived'])
print(data['Survived'])
print(data.describe())
print(data.groupby(['Sex']).mean())
print(data.groupby(['Survived','Sex','Pclass']).mean())
print(data.describe())
print(data['Age'].value_counts())
#data['Age'].hist()
#plt.bar(data['Sex'],data['Age'])
#sns.pairplot(data)
#sns.catplot(x='Survived', y='Age',data=data,hue='Sex' )
#sns.catplot(x='Survived', y='Age',data=data,hue='Pclass' )
#sns.boxplot(x='Survived', y='Age',data=data,hue='Pclass' )
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

data['Function']=data['Name'].str.split(" ", expand= True)[1]
print(data['Function'])
#print(data['Name'])
#data[['Gender','Name']]= data['Name'].str.split(". ", expand= True)
Title_Dictionary = {  "Capt":       "Officer",

  "Col":        "Officer",

  "Major":      "Officer",

    "Dr":         "Officer",

  "Rev":        "Officer",

  "Jonkheer":   "Royalty",

  "Don":        "Royalty",

  "Sir" :       "Royalty",

 "Lady" :      "Royalty",

"the Countess": "Royalty",

  "Dona":       "Royalty",

  "Mme":        "Miss",

  "Mlle":       "Miss",

  "Miss" :      "Miss",

  "Ms":         "Mrs",

  "Mr" :        "Mrs",

  "Mrs" :       "Mrs",

  "Master" :    "Master"

  }

data['Function']= data['Function'].map({  "Capt.":       "Officer",

  "Col.":        "Officer",

  "Major.":      "Officer",

    "Dr.":         "Officer",

  "Rev.":        "Officer",

  "Jonkheer.":   "Royalty",

  "Don.":        "Royalty",

  "Sir." :       "Royalty",

 "Lady." :      "Royalty",

"the Countess.": "Royalty",

  "Dona.":       "Royalty",

  "Mme.":        "Miss",

  "Mlle.":       "Miss",

  "Miss." :      "Miss",

  "Ms.":         "Mrs",

  "Mr." :        "Mrs",

  "Mrs." :       "Mrs",

  "Master." :    "Master"

  })
print(data['Function'])
sns.boxplot(x='Survived', y='Age',data=data,hue='Function' )

