import csv
import sys
import numpy as np 
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import data_handler   
# define model
class Vanilla_LSTM():
    
    def __init__(self, input_shape):
        self.model = Sequential()
        self.model.add(LSTM(50, activation='relu', input_shape=input_shape))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')
        self.n_features = 1
    
    def train(self, X, y):
        self.model.fit(X, y, epochs=10, verbose=1)

    def test(self, X, y):
        
        for i in range(len(X)):
            input = X[i].reshape((1,9,1))
            yhat = self.model.predict(input, verbose=1)
            print(y[i], yhat)
        # model.fit(X, y, epochs=200, verbose=1)

if __name__ == "__main__":
    # numpy.set_printoptions(threshold=sys.maxsize)
    time = data_handler.read_data("lob_data.csv","TIME")
    prices = data_handler.read_data("lob_data.csv", "MIC")
    
    # splitting data into chunks of 4
    steps = 9
    reshape = True
    X, y = data_handler.split_data(prices, steps, reshape)
    
    split_ratio = [9,1]
    train_X, test_X = data_handler.split_train_test_data(X, split_ratio)
    
    train_X = train_X.reshape((-1, steps, 1))
    test_X = test_X.reshape((-1, steps, 1))
    # print(train_X.shape)
    print(test_X.shape)
    
    train_y, test_y = data_handler.split_train_test_data(y, split_ratio)

    model = Vanilla_LSTM((steps,1))
    model.train(train_X, train_y)
    model.test(test_X, test_y)
    
        
    
    
    
    
   
    


