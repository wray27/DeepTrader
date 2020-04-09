import csv
import sys
import numpy as np

# read the data from CSV file  
# d_type 
# MID - mid prices, MIC - micro price, IMB - imbalances, SPR - spread


def normalize_data(x, max=0, min=0, train=True):
    if train:
        max = np.max(x)
        min = np.min(x)

    normalized = (x-min)/(max-min)
    return normalized


def normalize_data2(x):
    normalized = (2*(x-np.min(x))/(np.max(x)-np.min(x))) - 1
    return normalized

def standardize_data(x):
    standardized = (x - np.mean(x)) / np.std(x)
    return standardized

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


def read_data2(filename, d_type):
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
            elif d_type == "BID":
                data = np.append(data, float(row[5]))
            elif d_type == "ASK":
                data = np.append(data, float(row[6]))
            elif d_type == "TAR":
                data = np.append(data, float(row[9]))
            elif d_type == "OCC":
                data = np.append(data, float(row[8]))
            elif d_type == "DT":
                data = np.append(data, float(row[9]))
            else:
                data = np.append(data, float(row[0]))

    data = normalize_data(data)

    return data


def read_data3(filename, d_type):
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
            elif d_type == "BID":
                data = np.append(data, float(row[5]))
            elif d_type == "ASK":
                data = np.append(data, float(row[6]))
            elif d_type == "TAR":
                data = np.append(data, float(row[9]))
            else:
                data = np.append(data, float(row[0]))

    return data


def read_all_data(filename):
    data = {}
    
    with open(filename, "r") as f:
        f_data = csv.reader(f)
        data["TIME"] = np.array([])
        data["TYP"] = np.array([])
        data["LIM"] = np.array([])
        data["MID"] = np.array([])
        data["MIC"] = np.array([])
        data["IMB"] = np.array([])
        data["SPR"] = np.array([])
        data["BID"] = np.array([])
        data["ASK"] = np.array([])
        # data["TAR"] = np.array([])
        # data["OCC"] = np.array([])
        # data["DT"] = np.array([])
        # data["WMA"] = np.array([])

        for row in f_data:
            # print(row)
            data["TIME"] = np.append(data["TIME"],float(row[0]))
            data["TYP"] = np.append(data["TYP"], float(row[1]))
            data["LIM"] = np.append(data["LIM"], float(row[2]))
            data["MID"] = np.append(data["MID"],float(row[3]))
            data["MIC"] = np.append(data["MIC"], float(row[4]))
            data["IMB"] = np.append(data["IMB"],float(row[5]))
            data["SPR"] = np.append(data["SPR"], float(row[6]))
            data["BID"] = np.append(data["BID"], float(row[7]))
            data["ASK"] = np.append(data["ASK"], float(row[8]))
            
            # data["TIME"] = np.append(data["TIME"], float(row[0]))
            # data["TYP"] = np.append(data["TYP"], float(row[1]))
            # data["LIM"] = np.append(data["LIM"], float(row[1]))
            # data["MID"] = np.append(data["MID"], float(row[2]))
            # data["MIC"] = np.append(data["MIC"], float(row[3]))
            # data["IMB"] = np.append(data["IMB"], float(row[4]))
            # data["SPR"] = np.append(data["SPR"], float(row[5]))
            # data["BID"] = np.append(data["BID"], float(row[6]))
            # data["ASK"] = np.append(data["ASK"], float(row[7]))

            # data["TAR"] = np.append(data["TAR"], float(row[7]))
            # data["OCC"] = np.array(data["OCC"], float(row[8]))
            # data["DT"] = np.append(data["DT"], float(row[9]))
            # data["WMA"] = np.append(data["WMA"], float(row[10]))

        # for dataset in data:
        #     data[dataset] = normalize_data2(data[dataset])

                    
    temp = np.array([])
    temp = np.column_stack([data[d] for d in data])
  
    return temp

def read_data_from_multiple_files(no_files, no_features):
    
    X = np.array([[]])
    Y = np.array([])
    
    # retrieving data from multiple files
    for i in range(no_files):
        filename = f"./Data/Training/trial{(i+1):04}.csv"
        data = read_all_data(filename)
        transaction_prices = read_data3(filename, "TAR")
        X = np.append(X, data)
        Y = np.append(Y, transaction_prices)

    # reshaping input data
    X = np.reshape(X, (-1, no_features))


    return X, Y


def get_data(no_files, no_features):

    # obtaining data
    X, Y = read_data_from_multiple_files(no_files, no_features)
    # print(X.shape, Y.shape)
    # ratio of split as an array
    ratio = [9,1]

    # splitting train and test data for targets and input
    train_X, test_X = split_train_test_data(X, ratio)
    train_Y, test_Y = split_train_test_data(Y, ratio)

    # reshaping input to be correct
    train_X = np.reshape(train_X,(-1, no_features))
    test_X = np.reshape(test_X, (-1, no_features))

    train_max = np.array([float(0)]*(no_features + 1))
    train_min = np.array([float(0)]*(no_features + 1))

    # normalizing data
    # note: treating the test set the same way as the training set
    for c in range(no_features):
        # storing values used to scale
        train_max[c] = np.max(train_X[:, c])
        train_min[c] = np.min(train_X[:, c])
        
        # normalizing each feature using the only the training data to scale
        train_X[:, c] = normalize_data(train_X[:,c])
        test_X[:, c] = normalize_data(test_X[:, c], max=train_max[c], min=train_min[c], train=False)
        # print(np.max(train_X[:, c]), np.min(train_X[:, c]))

    # normalizing target data in the same way
    train_max[no_features] = np.max(train_Y)
    train_min[no_features] = np.min(train_Y)

    train_Y = normalize_data(train_Y)
    test_Y = normalize_data(test_Y, max=train_max[no_features], min=train_min[no_features], train=False)

    print(train_max)
    print(train_min)
    # print(train_X)
    # print(test_X)
    # print(train_Y)
    # print(test_Y)
    
    # reshaping input and target data for nn
    train_X = np.reshape(train_X, (-1, 1, no_features))
    train_Y = np.reshape(train_Y, (-1, 1))
    test_X = np.reshape(test_X, (-1, 1, no_features))
    test_Y = np.reshape(test_Y, (-1, 1))

    return train_X, train_Y, test_X, test_Y




# splitting data into input and output signal
# n_steps is the number of steps taken until a split occurs will have to formalize this with time steps
# for now is just for every n_steps, we have a y
def split_data(data, n_steps):

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
    
    
    A = A[:-1]
    A = np.reshape(A, (-1, n_steps,1))

    A = (A - np.mean(A)) / np.max(A)
    B = (B - np.mean(B)) / np.max(B)

    return A, B
def multi_split_data(data, x_steps, y_steps):
    
    A = np.array([])
    B = np.array([])

    step = 0
    add_A = True
    for d in np.nditer(data):

        if add_A:
            
            A = np.append(A, d)
            step += 1
            
            if step == x_steps: 
                add_A = False
                step = 0
        
        else:
           
            B = np.append(B, d)
            step += 1

            if step == y_steps:
                add_A = True
                step = 0

    
   
    A = A[:-1]
    A = np.reshape(A, (-1, x_steps,1))

    B = np.reshape(B, (-1, y_steps, 1))
    return A, B

# ratio is train first and then test  
def split_train_test_data(data, ratio):
    
    A = np.array([])
    B = np.array([])

    split_index = int( ratio[0] / (ratio[0] + ratio[1]) * len(data) )

    A = np.append(A, data[:split_index])
    B = np.append(B, data[split_index:])

    return A, B



if __name__ == "__main__":
    pass
