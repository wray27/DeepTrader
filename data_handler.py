import csv
import sys
import numpy as np
import matplotlib.pyplot as plt

# read the data from CSV file  
# d_type 
# MID - mid prices, MIC - micro price, IMB - imbalances, SPR - spread
def read_data(filename, d_type):
    data = np.array([])

    with open(filename, "r") as f:
        f_data = csv.reader(f)

        for row in f_data:
            # print(row)
            if d_type == "MID":
                data = np.append(data, float(row[1]))
            elif d_type == "MIC":
                data = np.append(data, float(row[2]))
            elif d_type == "IMB":
                data = np.append(data, float(row[3]))
            elif d_type == "SPR":
                    data = np.append(data, float(row[4]))
            else:
                data = np.append(data, float(row[0]))

    return data


# splitting data into input and output signal
# n_steps is the number of steps taken until a split occurs will have to formalize this with time steps
# for now is just for every n_steps, we have a y
def split_data(data, n_steps, reshape):

    A = np.array([])
    B = np.array([])

    step = 0
    for d in np.nditer(data):

        if step == n_steps + 1:
            A = np.append(A, d)
            step = 0
        else:
            B = np.append(B, d)
        step += 1
    
    if reshape:
        A = X[:-1]
        A = np.reshape(A, (-1, n_steps))

    return A, B

def time_series_plot():
    d_types = ["TIME","MID","MIC","IMB","SPR"]
    data = {}
    for d in d_types:
        data[d] = np.array([])
        data[d] = read_data("lob_data.csv", d)

    # for i in range(1, len(d_types)):
    for i in range(2, 3):
        plt.plot( data["TIME"], data[d_types[i]], label=d_types[i])
    
    plt.legend()
    plt.show()

if __name__ == "__main__":
    time_series_plot()
