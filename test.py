# Ejemplo de MLPregressor:
# https://www.programcreek.com/python/example/93778/sklearn.neural_network.MLPRegressor
# https://www.kaggle.com/ragnisah/eda-abalone-age-prediction 
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import  train_test_split

data = pd.read_csv('abalone.csv')
# cambiar de texto a datos
data['Sex'] = LabelEncoder().fit_transform(data['Sex'].tolist())
# calcular la edad con los anillos
data['Age'] = data['Rings'] + 1.5
# quitar los anillos
data.drop('Rings', axis = 1, inplace = True)

# separa la columna de la edad lo que se va a predecir
y = data['Age']
X = data.drop('Age', axis=1)

# Prueba = 20% y entranmiento 80%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# creación del modelo
model = MLPRegressor(hidden_layer_sizes=(5,),
                                activation='relu',
                                solver='adam',
                                learning_rate='adaptive',
                                max_iter=1000,
                                learning_rate_init=0.01,
                                alpha=0.01)

# entrenar el modelo
model.fit(X_train, y_train)

predi = model.predict(X_test)
score = model.score(X_test, predi)

print(score)