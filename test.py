# Ejemplo de MLPregressor:
# https://www.programcreek.com/python/example/93778/sklearn.neural_network.MLPRegressor
# https://www.kaggle.com/ragnisah/eda-abalone-age-prediction 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import  train_test_split
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import accuracy_score



data = pd.read_csv('abalone.csv')

# información de los datos
#data.info()

# cambiar de texto a datos
data['Sex'] = LabelEncoder().fit_transform(data['Sex'].tolist())
# calcular la edad con los anillos y mapear! 
def mapear(a):
    a+= 1.5 # se le suma 1.5 que es la edad
    if(a >= 9.5 and a <= 12.5):
        retornar = 'joven'
    elif(a < 9.5):
        retornar = 'infante'
    elif(a > 12.5):
        retornar = 'adulto'
    return retornar

rings = data['Rings'].tolist()
data['Age'] = list(map(mapear, rings))

# quitar los anillos
data.drop('Rings', axis = 1, inplace = True)

'''
# Grafica de la edad
ax = plt.subplots(1,1,figsize=(10,8))
sns.countplot('Age', data=data)
plt.title("Edad")
plt.show()
'''

'''
#historigrama de los datos
data.hist(edgecolor='black', linewidth=1.2, grid=False)
fig=plt.gcf()
plt.show()
'''
'''
sns.boxplot(data = data,width=0.5,fliersize=5)
plt.show()
'''
# heatmap
'''
numerical_features = data.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(20,7))
sns.heatmap(data[numerical_features].corr(), annot=True)
plt.show() 
'''

'''
sns.regplot(x='Diameter', y='Length', data=data)
sns.set(rc={'figure.figsize':(2,5)})
plt.show()
'''


# separa la columna de la edad lo que se va a predecir
y = data['Age']
X = data.drop('Age', axis=1)

# Prueba = 20% y entranmiento 80%
np.random.seed(1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")

y_train = LabelEncoder().fit_transform(y_train.tolist())
# creación del modelo
param_grid = [{
    'hidden_layer_sizes' : [(18, 3), (19, 3), (17, 3)], 
    'max_iter':[7000], 
    'solver': ['lbfgs'], 
    'activation' : ['tanh'], 
    'alpha': [0.08]
    }]
model = MLPClassifier()
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', iid=False)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

'''
model = MLPClassifier(alpha = 0.08, max_iter = 7000, activation = 'tanh', hidden_layer_sizes=(18, 3),solver="lbfgs")

# entrenar el modelo
model.fit(X_train, y_train)
# trata de predecir
Y_predi = model.predict(X_test)
# revisa el puntaje de la predicción
score = model.score(X_test, y_test)
print(score)
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, Y_predi))


def pintar(a):

    if(a == 'joven'):
        return 'r'
    if(a == 'infante'):
        return 'g'
    if(a == 'adulto'):
        return 'b'

lista = list(map(pintar, y_train))
lista2 = list(map(pintar, Y_predi))
plt.figure()
plt.scatter(X_train['Whole weight'], X_train['Diameter'], c=lista, label='real')
plt.show()
plt.figure()
plt.scatter(X_test['Whole weight'], X_test['Diameter'], c=lista2, label='Prediction')
plt.show()
'''