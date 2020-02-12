import csv
import sys
import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import data_handler
import data_visualizer
import NeuralNetwork

# Univariate LSTM 
class Vanilla_LSTM(NeuralNetwork.NeuralNetwork):
  
    def test(self, X, y, verbose):
        for i in range(len(X)):
        input = X[i].reshape((1,self.steps,1))
        yhat = self.model.predict(input, verbose=verbose)
        print(y[i], yhat[0][0]), np.mean(input[0]))
    
       
if __name__ == '__main__':
    steps = 59
    vanilla = Vanilla_LSTM((steps,1))
    reshape = True
    time = data_handler.read_data("./Data/lob_datatrial0001.csv", "TIME")
    prices = data_handler.read_data("./Data/lob_datatrial0001.csv", "MIC")
    X, y = data_handler.split_data(prices, steps, reshape)
    
    split_ratio = [9,1]
    train_X, test_X = data_handler.split_train_test_data(X, split_ratio)
    train_X = train_X.reshape((-1, steps, 1))
    test_X = test_X.reshape((-1, steps, 1))
    train_y, test_y = data_handler.split_train_test_data(y, split_ratio)
    
    vanilla.train(train_X, train_y, 200, verbose=1)
    vanilla.test(test_X, test_y, verbose=1)

    
