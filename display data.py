import math
import numpy as np
import pandas as pd
import graphviz as gz
import sklearn as sl


names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv('Pima_Indians_Diabetes.csv', names=names)

print(data)
print(data.describe())

import matplotlib.pyplot as plt
pd.options.display.mpl_style = 'default'
data.boxplot()

data.hist()

data.groupby('class').hist()

data.groupby('class').plas.hist(alpha=0.4)

from pandas.plotting import scatter_matrix
scatter_matrix(data, alpha=0.2, figsize=(6, 6), diagonal='kde')

