import keras
import tensorflow
import math
import sys
import pandas as pd
import numpy as np
from keras.models import load_model
from keras import layers, optimizers, losses, metrics

data = pd.read_csv(sys.argv[1], sep=',', header=None, usecols=[*range(1, 129)])
#print(data)

model = load_model('WindDenseNN.h5')

# summarize the structure of the model
model.summary()

result = model.predict(data, batch_size=32)
#print(result.shape)

actual_data = pd.read_csv('actual.csv', sep=',', header=None, usecols=[*range(1, 8)])
actual_data = np.array(actual_data)

MAE = 0
MSE = 0

for i in range(len(actual_data)):
    for j in range(len(actual_data[i])):
        MAE = MAE + abs(actual_data[i][j] - result[i][j])
        MSE = MSE + pow((actual_data[i][j] - result[i][j]), 2)

MAE = MAE/len(actual_data)
MSE = MSE/len(actual_data)

print("MAE = ", MAE)
print("MSE = ", MSE)
print("MAPE = ", np.mean(np.abs((actual_data - result) / np.mean(actual_data))) * 100)
