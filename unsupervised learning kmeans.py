'''K means and hiearchical clustering 
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


data = pd.read_csv('CC GENERAL.csv')
print(data.head)
print(data.columns)
print(data.describe)

#hiearchical clustering 

#print(data.dtypes)

data.drop(['CUST_ID'],axis=1,inplace=True)
data.dropna(axis=0,inplace=True)

print(data.dtypes)
print(data.head)
plt.figure(figsize=(12,12))
sns.heatmap(data.corr(),annot=True)
plt.show()

from sklearn.cluster import KMeans
'''
kmeans= KMeans(n_clusters=5,random_state=0)
kmeans.fit(data)
kmeans.predict(data)
print(kmeans.clusters_centers_)'''

SSE=[]
K=range(1,30)
for k in K :
    kmeans=KMeans(n_clusters=k)
    kmeans=kmeans.fit(data)
    SSE.append(kmeans.inertia_)
plt.plot(K,SSE, 'bx-')
plt.xlabel('number of clusters')
plt.ylabel('sum of squared distance')
plt.title('elbow method to determine k ')

kmeans= KMeans(n_clusters=3,random_state=0)
kmeans.fit(data)
kmeans.predict(data)

data['cluster_id']=kmeans.labels_

plt.figure(figsize=(10,6))
sns.scatterplot(data=data, x='ONEOFF_PURCHASES',y='PURCHASES',hue='cluster_id')
plt.title('clusters of purchases based on one off purchases and total purchases')
plt.show()

plt.figure(figsize=(10,6))
sns.scatterplot(data=data, x='CREDIT_LIMIT',y='PURCHASES',hue='cluster_id')
plt.title('clusters of purchases based on credit limit and total purchases')
plt.show()
