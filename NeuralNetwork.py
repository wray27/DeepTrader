import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# default is a univariate LSTM with one hidden layer that contaons 8 nodes
class NeuralNetwork():
    def __init__(self, input_shape):
        # inputs: A 3D tensor with shape[batch, timesteps, feature].
        
        self.input_shape = input_shape
        self.model = Sequential()
        self.steps = input_shape[0]
        self.model.add(LSTM(8, activation='relu', input_shape=input_shape))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', metrics=['accuracy'], loss='mae')
        self.n_features = self.input_shape[1]


    def train(self, X, y, epochs, verbose):
        self.model.fit(X, y, epochs=epochs, verbose=verbose)

    def test(self, X, y, verbose):
        for i in range(len(X)):
            input = X[i].reshape((1,self.steps,1))
            yhat = self.model.predict(input, verbose=verbose)
            print(y[i], yhat[0][0])
        
