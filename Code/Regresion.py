from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats
import seaborn as sns
from sklearn.model_selection import train_test_split

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


'''Funcions per aplicar la regressió'''
def mse(v1, v2):
    return ((v1 - v2)**2).mean()

def regression(x, y):
    # Creem un objecte de regressió de sklearn
    regr = LinearRegression()

    # Entrenem el model per a predir y a partir de x
    regr.fit(x, y)

    # Retornem el model entrenat
    return regr

def standarize(x_train):
    mean = x_train.mean(0)
    std = x_train.std(0)
    x_t = x_train - mean[None, :]
    x_t /= std[None, :]
    return x_t

x = data[:,:41]
y = data[:,41]

''' Dividim les dades que tenim en un conjunt d'entrenament i un conjunt de test '''

x_train, x_test, y_train, y_test = train_test_split(
                                        x,
                                        y,
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )

'''Fem la regressió per cada atribut, sense normalitzar'''

entreno = []
for i in range(x_train.shape[1]):
    entreno.append(regression(x_train[:,i].reshape(-1,1),y_train))
entreno = np.array(entreno)
predicciones = []
for i in range(x_test.shape[1]):
    predicciones.append(entreno[i].predict(x_test[:,i].reshape(-1,1)))


with open("../errores-cuadraticos.txt",'w') as s:
    for i, pred in enumerate(predicciones):
        s.write("Atributo " + str(i+1) + " : " + str(mse(standarize(pred.reshape(-1,1)),standarize(y_test.reshape(-1,1)))) + "\n")

#Generem grafiques per visualitzar la regressió
for i in range(x_test.shape[1]):
    plt.figure()
    ax = plt.scatter(x_test[:,i], y_test)
    plt.plot(x_test[:,i], predicciones[i], 'r')
    plt.savefig("../Grafiques/regresiones/no-normalizado/Atribut-" + str(i+1) + ".png")
