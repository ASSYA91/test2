import numpy as np 
#exercice 1 

x = 0
y = []
print (x)

for i in range (2000,3201):
    if (i % 7 == 0)&(i %5 != 0):
       y.append(i)
print (y)

#factoriel

def factoriel(n):
    if n == 0:
        return 1
    else:
        y=1
        for i in range (1,n+1):
            y= y*i
        return y
    
print (factoriel(5))

# dictionnaire des carrés
dict_1 = {}
def dictionnaire(n):
    for i in range (1, n+1):
        dict_1[i]= i**2
    print (dict_1)
print(dictionnaire(8))
#REMOVING A CHAR 

def cut(chaine,k):
    
    
    new_chaine = ""
    for i in range (0,len(chaine)):
        if i!= k:
            new_chaine = new_chaine+ chaine[i]
    print(new_chaine)

cut('tamim',1)


#numpy : question 5


array = np.array([[1,2],[3,4],[5,6]])

print(array)
print(array.shape)
liste_1= array.tolist()
print(liste_1)

#numpy : question 6
original_array_1= np.array([0,1,2])
print(original_array_1)
print(original_array_1.shape)

original_array_2= np.array([2,1,0])
cov= np.cov(original_array_1,original_array_2)
print(cov)

#numpy : question 7:square root 
C=50
H= 30
D= [18, 22, 24]
Q=[]
for i in range (0, len(D)):
    s= (2* C* D[i]/ H)**(0.5)
    Q.append(round(s))
print(Q)
        



    




    
    
    
    




        
        
    





        
    
        
        
    
        
        
    
    
