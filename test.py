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
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import  train_test_split
from sklearn.exceptions import ConvergenceWarning

data = pd.read_csv('abalone.csv')

# información de los datos
#data.info()

# cambiar de texto a datos
data['Sex'] = LabelEncoder().fit_transform(data['Sex'].tolist())
# calcular la edad con los anillos
data['Age'] = data['Rings'] + 1.5
# quitar los anillos
data.drop('Rings', axis = 1, inplace = True)

# Grafica de la edad
'''
ax = plt.subplots(1,1,figsize=(10,8))
sns.countplot('Age',data=data)
plt.title("Edad")
plt.show()
'''

#historigrama de los datos
'''
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")

# creación del modelo
model = MLPRegressor(hidden_layer_sizes=(5, ),
                                activation='tanh',
                                solver='adam',
                                learning_rate='adaptive',
                                max_iter=1000,
                                learning_rate_init=0.01,
                                alpha=0.001)
# entrenar el modelo
model.fit(X_train, y_train)
# trata de predecir
Y_predi = model.predict(X_test)
# revisa el puntaje de la predicción
score = model.score(X_test, y_test)
print(score)