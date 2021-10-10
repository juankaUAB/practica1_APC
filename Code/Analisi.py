from sklearn.datasets import make_regression
from sklearn.preprocessing import normalize as norm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats
import seaborn as sns

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

#Normalitzem les dades XD
data = norm(np.array(dataset.values), axis=0, norm='max')

x = data[:,:41]
y = data[:,41]

#Visualitzar valors buits
print("Hi han valors buits/nulls? " + str(any(dataset.isnull().sum()) != 0))

#Veure estadistiques de la BBDD
#print(dataset.describe())

#Generem grafiques per tots els atributs d'entrada
for i in range(x.shape[1]):
    plt.figure()
    ax = plt.scatter(x[:,i],y)
    plt.savefig("../Grafiques/punts/" + "Atribut-" + str(i + 1) + ".png")

''' Generació de grafiques de CORRELACIÓ per cada tipus de variable, amb els 4 quarts del any
    junts o els 4 quarts per separat (q1, q2, q3 i q4) '''
    
#Generem grafica de correlació entre variables (CQRSA en tots els quarts del any junts)
dataset1 = dataset.drop([str("f" + str(x)) for x in range(11,41)],axis=1)
correlacio = dataset1.corr()
plt.figure()
ax = sns.heatmap(correlacio, annot=True)
plt.savefig("../Grafiques/correlacio/correlacio_CQRSA_allQuarters.png")

#Generem grafica de correlació entre variables (CQR en tots els quarts del any junts)
dataset1 = dataset.drop([str("f" + str(x)) for x in range(1,11)],axis=1)
dataset1 = dataset1.drop([str("f" + str(x)) for x in range(21,41)],axis=1)
correlacio = dataset1.corr()
plt.figure()
ax = sns.heatmap(correlacio, annot=True)
plt.savefig("../Grafiques/correlacio/correlacio_CQR_allQuarters.png")

#Generem grafica de correlació entre variables (VNBQRSA en tots els quarts del any junts)
dataset1 = dataset.drop([str("f" + str(x)) for x in range(1,21)],axis=1)
dataset1 = dataset1.drop([str("f" + str(x)) for x in range(31,41)],axis=1)
correlacio = dataset1.corr()
plt.figure()
ax = sns.heatmap(correlacio, annot=True)
plt.savefig("../Grafiques/correlacio/correlacio_VNBQRSA_allQuarters.png")

#Generem grafica de correlació entre variables (VNBQR en tots els quarts del any junts)
dataset1 = dataset.drop([str("f" + str(x)) for x in range(1,31)],axis=1)
correlacio = dataset1.corr()
plt.figure()
ax = sns.heatmap(correlacio, annot=True)
plt.savefig("../Grafiques/correlacio/correlacio_VNBQR_allQuarters.png")


