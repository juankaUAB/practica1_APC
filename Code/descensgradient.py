from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# Funcio per a llegir dades en format csv
def load_dataset(path, path1, path2, path3):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    dataset1 = pd.read_csv(path1, header=0, delimiter=',')
    dataset2 = pd.read_csv(path2, header=0, delimiter=',')
    dataset3 = pd.read_csv(path3, header=0, delimiter=',')
    
    dataset = pd.concat([dataset, dataset1, dataset2, dataset3],ignore_index=True)
    return dataset

def standarize(x_train):
    mean = x_train.mean(0)
    std = x_train.std(0)
    x_t = x_train - mean[None, :]
    x_t /= std[None, :]
    return x_t

#Carregar dades de la BBDD
dataset = load_dataset("../BDD/q1.csv","../BDD/q2.csv","../BDD/q3.csv","../BDD/q4.csv")
data = standarize(dataset.values)

def mse(v1, v2):
    return ((v1 - v2)**2).mean()


'''Funcions per fer el descens del gradient'''

def coste(x, y, a, b):
    m = len(x)
    error = 0.0
    for i in range(m):
        hipotesis = a+b*x[i]
        error +=  (y[i] - hipotesis) ** 2
    return error / (2*m)

def descenso_gradiente(x, y, a, b, alpha, epochs):
    m = len(x)
    hist_coste = []
    for ep in range(epochs):
        b_deriv = 0
        a_deriv = 0
        for i in range(m):
            hipotesis = a+b*x[i]
            a_deriv += hipotesis - y[i]
            b_deriv += (hipotesis - y[i]) * x[i]
            hist_coste.append(coste(x, y, a, b))
        a -= (a_deriv / m) * alpha
        b -= (b_deriv / m) * alpha
        
    return a, b, hist_coste


'''Fem la regressió per cada atribut, sense normalitzar'''
plt.figure()


'''descens de gradient per l'atribut 22'''
x = data[:,21]
y = data[:,41]

w0, w1, cost_historial = descenso_gradiente(x,y,1,1,0.05,1000)

recta = w1*x + w0
plt.scatter(x,y)
plt.plot(x, recta)


valors22 = []
for valor in data[:,21]:
    valors22.append(valor*w1 + w0)
error1 = mse(valors22, data[:,21])
plt.savefig("../Grafiques/descensgradient/""atribut22.png")
plt.clf()


'''descens de gradient per l'atribut 31'''
x = data[:,30]
y = data[:,41]

w0, w1, cost_historial = descenso_gradiente(x,y,1,1,0.05,1000)

recta2 = w1*x + w0
plt.scatter(x,y)
plt.plot(x, recta2)

valors31 = []
for valor in data[:,30]:
    valors31.append(valor*w1 + w0)
error2 = mse(valors31, data[:,30])

plt.savefig("../Grafiques/descensgradient/""atribut31.png")
plt.clf()

'''descens de gradient per l'atribut 32'''
x = data[:,31]
y = data[:,41]

w0, w1, cost_historial = descenso_gradiente(x,y,1,1,0.05,1000)

recta3 = w1*x + w0
plt.scatter(x,y)
plt.plot(x, recta3)

valors32 = []
for valor in data[:,31]:
    valors32.append(valor*w1 + w0)
error3 = mse(valors32, data[:,31])

plt.savefig("../Grafiques/descensgradient/""atribut32.png")
plt.clf()

'''descens de gradient per l'atribut 38'''
x = data[:,37]
y = data[:,41]

w0, w1, cost_historial = descenso_gradiente(x,y,1,1,0.05,1000)

recta4 = w1*x + w0
plt.scatter(x,y)
plt.plot(x, recta4)

valors15 = []
for valor in data[:,37]:
    valors15.append(valor*w1 + w0)
error4 = mse(valors15, data[:,37])

plt.savefig("../Grafiques/descensgradient/""atribut38.png")
plt.clf()







