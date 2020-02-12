import csv
import sys
import numpy as np
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


def read_all_data(filename):
    data = {}

    with open(filename, "r") as f:
        f_data = csv.reader(f)
        data["TIME"] = np.array([])
        data["MID"] = np.array([])
        data["MIC"] = np.array([])
        data["IMB"] = np.array([])
        data["SPR"] = np.array([])

        for row in f_data:
            data["TIME"] = np.append(data["TIME"],float(row[0]))
            data["MID"] = np.append(data["MID"],float(row[1]))
            data["MIC"] = np.append(data["MIC"],float(row[2]))
            data["IMB"] = np.append(data["IMB"],float(row[3]))
            data["SPR"] = np.append(data["SPR"],float(row[4]))
 
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
            B = np.append(B, d)
            step = 0
        else:
            A = np.append(A, d)
        step += 1
    
    if reshape:
        A = A[:-1]
        A = np.reshape(A, (-1, n_steps,1))

    return A, B

# ratio is train first and then test  
def split_train_test_data(data, ratio):
    
    A = np.array([])
    B = np.array([])

    split_index = int( ratio[0] / (ratio[0] + ratio[1]) * len(data) )
    # print(split_index)

    A = np.append(A, data[:split_index])
    B = np.append(B, data[split_index:])

    return A, B



if __name__ == "__main__":
    pass
