from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats

# Visualitzarem només 3 decimals per mostra
pd.set_option('display.float_format', lambda x: '%.3f' % x)

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

x = data[:,:10]
y = data[:,41]

#Visualitzar valors buits
print(any(dataset.isnull().sum()) != 0)

#Veure estadistiques de la BBDD
print(dataset.describe())

#Visualitzar el ultim atribut (42)
#plt.figure()
#ax = plt.scatter(x[:,0],y)
