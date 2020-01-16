import keras
import tensorflow
import math
import sys
import csv
import pandas as pd
import numpy as np
from pandas import DataFrame
from keras.models import load_model
from keras import layers, optimizers, losses, metrics

data = pd.read_csv(sys.argv[1], sep=',', header=None, usecols=[*range(1, 129)])
timestamps = pd.read_csv(sys.argv[1], sep=',', header=None, usecols=[*range(0, 1)])
timestamps = np.array(timestamps)

model = load_model('WindDenseNN.h5')

new_model = keras.Sequential()

new_model.add(model.layers[0])

new_model.compile(optimizer=optimizers.RMSprop(0.01), loss=losses.CategoricalCrossentropy(), metrics=[metrics.CategoricalAccuracy()])

result = new_model.predict(data, batch_size=32)

final = np.append(timestamps, result, axis=1)

with open("cluster/repres_output.csv","w") as my_csv:
    csvWriter = csv.writer(my_csv, delimiter=',')
    csvWriter.writerows(final)
