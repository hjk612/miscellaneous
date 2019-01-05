import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
import matplotlib.pyplot as plt

os.chdir('/Users/Hatim/Desktop/ivani/')

data = pd.read_csv('sample_data.txt', sep = ', ', engine = 'python')
data['Timestamp'] = data['Timestamp'].map(lambda x: x.lstrip('[').rstrip(']'))
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
mapping = {True: 1, False: 0}
data = data.applymap(lambda s: mapping.get(s) if s in mapping else s)
data['time_diff'] = (data['Timestamp'] - data['Timestamp'].shift()).dt.total_seconds()
data.dropna(inplace = True)
np.random.seed(42)

train_size = int(len(data)*0.7)

train = data.iloc[:train_size]
test = data.iloc[train_size:]
X_train = train[['Stream1', 'Stream2', 'Stream4']].values
y_train = train['Truth'].values

X_test = test[['Stream1', 'Stream2', 'Stream4']].values
y_test = test['Truth'].values

lookback = 500

def create_dataset(X, y, lookback):
    dataX, dataY = [], []
    for i in range(len(X)-lookback-1):
        a = X[i:(i+lookback)]
        dataX.append(a)
        dataY.append(y[i + lookback])
    return np.array(dataX), np.array(dataY)

X_train_seq, y_train_seq = create_dataset(X_train, y_train, lookback)
X_test_seq, y_test_seq = create_dataset(X_test, y_test, lookback)

model = Sequential()
model.add(LSTM(100, input_shape=(lookback, 3)))
model.add(Dropout(0.7))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train_seq, y_train_seq, epochs=5)
_, test_accuracy = model.evaluate(X_test_seq, y_test_seq, verbose=0)

yhat = model.predict_classes(X_test_seq, verbose=0)
plt.plot(yhat.flatten())
plt.plot(y_test_seq)
plt.plot(data.Stream1 | data.Stream2 | data.Stream3)
