import keras
import tensorflow
import math
import sys
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from keras import layers, optimizers, losses, metrics

data = pd.read_csv(sys.argv[1], sep=',', header=None, usecols=[*range(1, 129)])
#print(data)

model = load_model('WindDenseNN.h5')

# summarize the structure of the model
model.summary()

print(model.layers[1])
