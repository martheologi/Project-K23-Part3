import keras
import tensorflow
import math
import sys
import csv
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from keras import layers, optimizers, losses, metrics

data = pd.read_csv(sys.argv[1], sep=',', header=None, usecols=[*range(1, 129)])
timestamps = pd.read_csv(sys.argv[1], sep=',', header=None, usecols=[*range(0, 1)])
timestamps = np.array(timestamps)

model = load_model('WindDenseNN.h5')

# summarize the structure of the model
#model.summary()

result = model.predict(data, batch_size=32)

actual_data = pd.read_csv('actual.csv', sep=',', header=None, usecols=[*range(1, 8)])
actual_data = np.array(actual_data)

MAE = mean_absolute_error(actual_data, result)
MSE = mean_squared_error(actual_data, result)
MAPE = np.mean(np.abs((actual_data - result) / np.mean(actual_data))) * 100

print("MAE = ", MAE)
print("MSE = ", MSE)
print("MAPE = ", MAPE)

output_file = open('predict_output.csv', 'w')
errors = '%s%f%s%f%s%f \n' % ("MAE = ", MAE, " MAPE = ", MAPE, " MSE = ", MSE)
output_file.write(errors)
output_file.close()

final = np.append(timestamps, result, axis=1)

with open("predict_output.csv","a+") as my_csv:
    csvWriter = csv.writer(my_csv, delimiter=',')
    csvWriter.writerows(final)
