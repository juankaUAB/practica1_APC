import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats
import seaborn as sns

#Establir els titols de cada grafica
title_x = []
g = open("../titulos_columnas.txt",'r')
for linea in g:
    title_x.append(linea)
g.close()
title_x = np.array(title_x)
title_y = np.array(["Numero de turistas"])

# Funcio per a llegir dades en format csv
def load_dataset(path, path1, path2, path3):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    dataset1 = pd.read_csv(path1, header=0, delimiter=',')
    dataset2 = pd.read_csv(path2, header=0, delimiter=',')
    dataset3 = pd.read_csv(path3, header=0, delimiter=',')
    
    dataset = pd.concat([dataset, dataset1, dataset2, dataset3],ignore_index=True)
    return dataset

#Carregar dades de la BBDD
dataset = load_dataset("../BDD/q1.csv","../BDD/q2.csv","../BDD/q3.csv","../BDD/q4.csv")
data = dataset.values

'''Calculem el coeficient de variació per cada variable'''
desviaciones = np.std(data,axis=0)
with open("../desviaciones.txt",'w') as d:
    for i, des in enumerate(desviaciones):
        d.write("Atributo " + str(i+1) + " : " + str(des) + "\n")
        d.write("----------------------------\n")

#Dividim les dades en la entrada i la sortida
x = data
y = data[:,41]

#Generem grafiques de dispersió i histogrames per tots els atributs d'entrada
for i in range(x.shape[1]):
    if i != 41:
        plt.figure()
        plt.xlabel(title_x[i])
        plt.ylabel(title_y[0])
        ax = plt.scatter(x[:,i],y)
        plt.savefig("../Grafiques/punts/" + "Atribut-" + str(i + 1) + ".png")
    plt.figure()
    if (i == 41):
        plt.xlabel(title_y[0])
    else:
        plt.xlabel(title_x[i])
    
    sns.histplot(
        data    = x[:,i],
        kde     = True,
        line_kws= {'linewidth': 2},
        color = 'g',
        alpha   = 0.3,
    )
    plt.savefig("../Grafiques/histogrames/" + "Atribut-" + str(i + 1) + ".png")

'''Apliquem un test de normalitat (el de Shapiro) a cadascuna de les variables per determinar 
quines ens seran utils (segueixen una distribuicio normal)'''
resultats = []
for i in range(x.shape[1]):
    resultats.append(scipy.stats.shapiro(x[:,i]))
resultats = np.array(resultats)
    
with open("../results_shapiroTest/resultados.txt",'w') as f:
    f.write(" - TEST DE SHAPIRO - \n")
    f.write("---------------------\n")
    for k, res in enumerate(resultats):
        f.write("Atributo " + str(k+1) + " : Estadistico: " + str(res[0]) + "   |   P-Valor: " + str(res[1]) + "\n")
        if res[1] < 0.05:
            f.write("Se puede rechazar la hipotesis de que los datos de distribuyen de forma normal\n")
        else:
            f.write("No se puede rechazar la hipotesis de que los datos de distribuyen de forma normal\n")
        f.write("-----------------------------------------------------------------------------\n")
    
    

''' Generació de grafiques de CORRELACIÓ per cada tipus de variable, amb els 4 quarts del any
    junts o els 4 quarts per separat (q1, q2, q3 i q4) '''
    
#Generem grafica de correlació entre variables (CQRSA en tots els quarts del any junts)
dataset1 = dataset.drop([str("f" + str(x)) for x in range(11,41)],axis=1)
correlacio = dataset1.corr(method="pearson")
plt.figure()
plt.title("CQRSA")
ax = sns.heatmap(correlacio, annot=True)
plt.savefig("../Grafiques/correlacio/correlacio_CQRSA_allQuarters.png")

#Generem grafica de correlació entre variables (CQR en tots els quarts del any junts)
dataset1 = dataset.drop([str("f" + str(x)) for x in range(1,11)],axis=1)
dataset1 = dataset1.drop([str("f" + str(x)) for x in range(21,41)],axis=1)
correlacio = dataset1.corr(method="pearson")
plt.figure()
plt.title("CQR")
ax = sns.heatmap(correlacio, annot=True)
plt.savefig("../Grafiques/correlacio/correlacio_CQR_allQuarters.png")

#Generem grafica de correlació entre variables (VNBQRSA en tots els quarts del any junts)
dataset1 = dataset.drop([str("f" + str(x)) for x in range(1,21)],axis=1)
dataset1 = dataset1.drop([str("f" + str(x)) for x in range(31,41)],axis=1)
correlacio = dataset1.corr(method="pearson")
plt.figure()
plt.title("VNBQRSA")
ax = sns.heatmap(correlacio, annot=True)
plt.savefig("../Grafiques/correlacio/correlacio_VNBQRSA_allQuarters.png")

#Generem grafica de correlació entre variables (VNBQR en tots els quarts del any junts)
dataset1 = dataset.drop([str("f" + str(x)) for x in range(1,31)],axis=1)
correlacio = dataset1.corr(method="pearson")
plt.figure()
plt.title("VNBQR")
ax = sns.heatmap(correlacio, annot=True)
plt.savefig("../Grafiques/correlacio/correlacio_VNBQR_allQuarters.png")


