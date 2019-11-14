# Ejemplo de MLPregressor:
# https://www.programcreek.com/python/example/93778/sklearn.neural_network.MLPRegressor
# https://www.kaggle.com/ragnisah/eda-abalone-age-prediction 
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor

data = pd.read_csv('abalone.csv')
# cambiar de texto a datos
data['Sex'] = LabelEncoder().fit_transform(data['Sex'].tolist())
# calcular la edad con los anillos
data['Age'] = data['Rings'] + 1.5
# quitar los anillos
data.drop('Rings', axis = 1, inplace = True)


print(data.head())