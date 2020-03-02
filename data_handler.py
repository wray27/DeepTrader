import csv
import sys
import numpy as np

# read the data from CSV file  
# d_type 
# MID - mid prices, MIC - micro price, IMB - imbalances, SPR - spread


def normalize_data(x):
    normalized = (x-np.min(x))/(np.max(x)-np.min(x))
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
                data = np.append(data, float(row[7]))
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
                data = np.append(data, float(row[7]))
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
        data["BID"] = np.array([])
        data["ASK"] = np.array([])
        # data["TAR"] = np.array([])
        # data["OCC"] = np.array([])
        data["DT"] = np.array([])

        for row in f_data:
            data["TIME"] = np.append(data["TIME"],float(row[0]))
            data["MID"] = np.append(data["MID"],float(row[1]))
            data["MIC"] = np.append(data["MIC"],float(row[2]))
            data["IMB"] = np.append(data["IMB"],float(row[3]))
            data["SPR"] = np.append(data["SPR"], float(row[4]))
            data["BID"] = np.append(data["BID"], float(row[5]))
            data["ASK"] = np.append(data["ASK"], float(row[6]))
            # data["TAR"] = np.append(data["TAR"], float(row[7]))
            # data["OCC"] = np.array(data["OCC"], float(row[8]))
            data["DT"] = np.append(data["DT"], float(row[9]))

        for dataset in data:
            data[dataset] = normalize_data(data[dataset])

        
                
    temp = np.array([])
    temp = np.vstack([data[d] for d in data])
    
    return temp

def read_data_from_multiple_files():
     
    arr = []
    arr2 = []
    for i in range(9):
        filename = "./Data/trial000" + str(i+1) + '.csv'
        data = read_all_data(filename)
        
        transaction_prices = read_data2(filename, "TAR")
        occurrences = read_data2(filename, "OCC")
        
        labels = np.column_stack((transaction_prices, occurrences))

        arr.append(data)
        arr2.append(labels)
      

    train = np.hstack([arr[i] for i in range(len(arr))])
    labels = np.hstack([arr2[i] for i in range(len(arr2))])
    
   
    train = np.reshape(train, (-1, 1, 8))
    labels = np.reshape(labels, (-1, 2))

    return train, labels


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
    # print(split_index)

    A = np.append(A, data[:split_index])
    # train_mean = np.mean(A)
    # train_std = np.std(A)
    # A = (A - train_mean) / train_std

    B = np.append(B, data[split_index:])
    
    print("shape of normalized data: ",A.shape)
    return A, B



if __name__ == "__main__":
    pass
