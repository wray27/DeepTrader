import csv
import sys
import numpy as np 
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import data_handler   
# define model
def Vanilla_LSTM(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    return model

def train(X, y, model):
    # print(X.shape)
    # print(y.shape)
    # X = np.reshape(X,(-1,4,1))
    # model.fit(X, y, epochs=200, verbose=1)
    pass


def test(X, y, model):
    pass
    # model.fit(X, y, epochs=200, verbose=1)

if __name__ == "__main__":
    # numpy.set_printoptions(threshold=sys.maxsize)
    time = data_handler.read_data("lob_data.csv","TIME")
    prices = data_handler.read_data("lob_data.csv", "MIC")
    
    
    # splitting data into chunks of 4
    steps = 4
    reshape = True
    X, y = data_handler.split_data(prices, steps, reshape)
   

    
    
    
    # print(train_X.shape, test_X.shape)
    # print(train_y.shape, test_y.shape)

    model = Vanilla_LSTM((1,4))
    train(train_X, train_y, model)
    # test(test_X, test_y, model)
        
    
    
    
    
   
    


