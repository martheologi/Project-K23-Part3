import keras
import pandas as pd
import numpy as np
from keras.models import load_model
from keras import layers, optimizers, losses, metrics

data = pd.read_csv('nn_representations.csv', sep=',', header=None, usecols=[*range(1, 129)])
print(data)

model = load_model('WindDenseNN.h5')

# summarize the structure of the model
model.summary()

result = model.predict(data, batch_size=64)
print(result.shape)
print(result)
