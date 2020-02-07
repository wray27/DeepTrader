
#  The intention of this file is to read time series of the microprices after each data and make a prediction ogf its mogemeent
import csv
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




if __name__ == "__main__":

    time, prices = read_marketprices("market_prices.csv", "MIC")

    print(prices)
