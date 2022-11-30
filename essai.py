
print("Welcome to python world! Let's start coding'")
def e_potentiel(masse, hauteur, g=9.81):
    E= masse * hauteur * g
    print(E,"joules")
        
    return E

resultat=e_potentiel(masse=50, hauteur=2, g=9.81) 



def signe(x,y):
    if (x > 0) &  (y>x):
        print (x,y,"positif")
    elif x==0:
        print (x,"nul")
    else:
        print (x,"négatif")
print(signe(0,6))

for i in range(10):
    print("bonjour")




    


    
    
















