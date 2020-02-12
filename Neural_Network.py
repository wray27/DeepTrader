import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense


class NeuralNetwork():
    def __init__(self, input_shape, hidden_layers):
        # inputs: A 3D tensor with shape[batch, timesteps, feature].
        
        self.input_shape = input_shape
        self.model = Sequential()
        self.steps = input_shape[0]
        
        self.model.add(LSTM(hidden_layers, activation='relu', input_shape=input_shape))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', metrics=['accuracy'], loss='mae')
        self.n_features = self.input_shape[2]


    def train(self, X, y, epochs, verbose):
        self.model.fit(X, y, epochs=epochs, verbose=verbose)

    def test(self, X, y):
        for i in range(len(X)):
            input = X[i].reshape((1,self.steps,1))
            yhat = self.model.predict(input, verbose=1)
            print(y[i], yhat[0][0])
        
