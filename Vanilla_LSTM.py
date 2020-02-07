
#  The intention of this file is to read time series of the microprices after each data and make a prediction ogf its mogemeent
import numpy
import csv
import sys
import numpy as np 
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# type is MID or MIC for mid and micro prices
def read_marketprices(filename, p_type):
    
    time = np.array([])
    prices = np.array([])
    
    with open(filename, "r") as f:
        data = csv.reader(f)

        for row in data:
            # print(row)
            time = np.append(time, float(row[0]))
            if p_type == "MID":
                prices = np.append(prices, float(row[1]))
            elif p_type == "MIC":
                prices = np.append(prices, float(row[2]))
    
    return time, prices

# splitting data into input and output signal
# n_steps is the number of steps taken until a split occurs will have to formalize this with time steps
# for now is just for every n_steps, we have a y  
def split_data(data, n_steps, split_type):
    
    X = np.array([[]])
    y = np.array([])
    
    step = 0
    for d in np.nditer(data):
        
        if step == n_steps + 1:
            y = np.append(y, d)
            step = 0
        else:
            X = np.append(X, d)
        step += 1
    
    
    X = X[:-1]
    X  = np.reshape(X, (-1,n_steps))

    return X,y

    
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
    time, prices = read_marketprices("market_prices.csv", "MIC")
    # splitting data into chunks of f
    X, y = split_data(prices, 4, True)
   

    train_X, test_X = 
    train_y, test_y = 
    
    
    print(train_X.shape, test_X.shape)
    print(train_y.shape, test_y.shape)

    model = Vanilla_LSTM((1,4))
    train(train_X, train_y, model)
    # test(test_X, test_y, model)
        
    
    
    
    
   
    


