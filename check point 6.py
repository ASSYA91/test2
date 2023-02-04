from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn import metrics
import csv
import random
import math 
import operator 


#with open("iris.data.txt",'r') as csvfile
#lines= csv.reader(csvfile)
#read_file= pd.read_csv("iris.data.txt")
#iris= read_file.to_csv('iris.data.csv')
#print(iris.coloumn)
with open('iris.data.txt', 'r') as csvfile:
    lines = csv.reader(csvfile)
    for row in lines :
        print(','.join(row))
def loadDataset(filename,split,trainingSet=[],testSet=[]):
    with open(filename,'r') as csvfile:
        lines=csv.reader(csvfile)
        dataset=list(lines)
        for x in range (len(dataset)-1):
            for y in range (4):
                dataset[x][y]=float(dataset[x][y])
                if random.random()< split: 
                    trainingSet.append(dataset[x])
                else:
                    testSet.append(dataset[x])
    return trainingSet,testSet
       

trainingSet,testSet=loadDataset('iris.data.txt', 0.66,trainingSet=[],testSet=[])

print('train:'+repr(len(trainingSet)))
print('test:'+ repr(len(testSet)))
# il faut toujours retourner mon vvariable souhaité dans la fonction puis le faire appeler dans une variable 


def euclideanDistance(instance_2,instance_1,length):
    dist=0
    for x in range (length):
        dist += pow((instance_2[x] -instance_1[x]), 2)
    return math.sqrt(dist)
#distance= euclideanDistance([2,2,2,'a'], [4,4,4,'b'], 3)
#print('distance',repr(distance))



#neighbours 

def getNeighbors(trainingSet,testInstance,k):
    distances=[]
    length=len(testInstance)-1
    for x in range (len(trainingSet)):
        dist= euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x],dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors=[]
    for x in range (k):
        neighbors.append(distances[x][0])
        return neighbors
        
#trainingSet=[[2,2,2,'a'],[4,4,4,'b']]
#testInstance=[5,5,5]
#k=1
#neighbors=getNeighbors(trainingSet, testInstance, k)
#print('neighbors are : ',neighbors)


# choose the classes 
def getResponse(neighbors):
    classVotes={}
    for x in range(len(neighbors)):
        response= neighbors[x][-1]
        if response in classVotes:
            classVotes[response]+= 1
        else:
          classVotes[response]=1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]



#neighbors = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]

#response = getResponse(neighbors)

#print(response)
#it doesn't work the example



def getAccuracy(testSet,predictions):
    correct=0
    for x in range(len(testSet)):
        if testSet[x][-1]== predictions[x]:
            correct +=1
    return(correct/float(len(testSet)))*100
testSet = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]

predictions = ['a', 'a', 'a']

accuracy = getAccuracy(testSet, predictions)

print(accuracy)

def main():
    
    trainingSet,testSet=loadDataset('iris.data.txt', 0.8,trainingSet=[],testSet=[])

    print('trainset:'+repr(len(trainingSet)))
    print('testset:'+repr(len(testSet)))#c 'est quoi repr
    predictions=[]
    k= 3 
    for x in range(len(testSet)):
        neighbors=getNeighbors(trainingSet, testSet[x], k)
        result=getResponse(neighbors)
        predictions.append(result)
    accuracy=getAccuracy(testSet, predictions)
    print('Accuracy:',accuracy)
    
main()


        


    
    
    
  
    
    

    


        




  
                

                    
                
                    

