
#  The intention of this file is to read time series of the microprices after each data and make a prediction ogf its mogemeent
import numpy
import csv
import sys
import numpy as np 

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
def split_data(data, n_steps):
    X = np.array([[]])
    y = np.array([])
    
    step = 0
    for d in np.nditer(data):
        
        if step == n_steps:
            y = np.append(y, d)
            step = 0
        else:
            X = np.append(X, d)
        step += 1

    return X,y
        

if __name__ == "__main__":
    numpy.set_printoptions(threshold=sys.maxsize)
    time, prices = read_marketprices("market_prices.csv", "MIC")
    X, y = split_data(prices, 4)
   
    print(X.shape)
    print(y.shape)


