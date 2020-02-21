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

# Univariate LSTM used to predict a single next step in time series data
class Vanilla_LSTM(NeuralNetwork.NeuralNetwork):
    def __init__(self, input_shape, filename):
        # inputs: A 3D tensor with shape[batch, timesteps, feature].

        self.input_shape = input_shape
        self.model = Sequential()
        self.steps = input_shape[0]
        self.model.add(LSTM(8, activation='relu', input_shape=input_shape))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', metrics=['accuracy'], loss='mae')
        self.n_features = self.input_shape[1]
        self.filename = filename

    def test(self, X, y, verbose):
        preds = np.array([])
        baseline = np.array([])
        
        for i in range(len(X)):
            input = X[i].reshape((1,self.steps,1))
            yhat = self.model.predict(input, verbose=verbose)
            preds = np.append(preds, yhat[0][0])
            baseline = np.append(baseline, np.mean(input[0]))
            print(y[i], preds[i], baseline[i])

        # print(len(y), len(preds), len(baseline))
        data_visualizer.accuracy_plot(y, preds, baseline)
        
    def run_all(self):
       
        time=data_handler.read_data("./Data/lob_data.csv", "TIME")
        prices=data_handler.read_data("./Data/lob_data.csv", "MIC")
        X, y=data_handler.split_data(prices, self.steps)

        split_ratio=[9, 1]
        train_X, test_X=data_handler.split_train_test_data(X, split_ratio)
        train_X=train_X.reshape((-1, self.steps, 1))
        test_X=test_X.reshape((-1, self.steps, 1))
        train_y, test_y=data_handler.split_train_test_data(y, split_ratio)

        self.train(train_X, train_y, 200, verbose = 1)
        self.test(test_X, test_y, verbose = 1)
        self.save()

    def run_all2(self):
        
        for i in range(9):
            print("epoch " + str(i+1) + " out of 9")
            prices = data_handler.read_data("./Data/lob_datatrial000" + str(i+1) + ".csv", "MIC")
            X, y = data_handler.split_data(prices, self.steps)
            self.train(X, y, 100, 0)
        
        time = data_handler.read_data("./Data/lob_data.csv", "TIME")
        prices = data_handler.read_data("./Data/lob_data.csv", "MIC")
        X, y = data_handler.split_data(prices, self.steps)
        self.test(X, y, verbose=1)


class MultiVanilla_LSTM(NeuralNetwork.NeuralNetwork):
    def __init__(self, input_shape, out_steps, filename):
        # inputs: A 3D tensor with shape[batch, timesteps, feature].

        self.input_shape = input_shape
        self.model = Sequential()
        self.steps = input_shape[0]
        self.out_steps = out_steps
        self.model.add(LSTM(24,  activation='relu', input_shape=input_shape))
        # self.model.add(LSTM(8,  return_sequences=True, activation='relu'))
        # self.model.add(LSTM(6, activation='relu'))
        self.model.add(Dense(self.out_steps))
        self.model.compile(optimizer='adam', metrics=['accuracy'], loss='mae')
        self.n_features = self.input_shape[1]
        self.filename = filename

    def test(self, X, y, verbose):
        preds = np.array([])
        baseline = np.array([])
        actual = np.array([])

        for i in range(len(X)):
            input = X[i].reshape((1, self.steps, 1))
            yhat = self.model.predict(input, verbose=verbose)
            preds = np.append(preds, yhat[0][self.out_steps - 1])
            baseline = np.append(baseline, np.mean(input[0]))
            actual = np.append(actual, y[i][self.out_steps - 1])
            
            print(actual[i], preds[i], baseline[i])

        # print(len(y), len(preds), len(baseline))
        data_visualizer.accuracy_plot(actual, preds, baseline)

    def run_all(self):
    
        time = data_handler.read_data("./Data/lob_datatrial0001.csv", "TIME")
        prices = data_handler.read_data("./Data/lob_datatrial0001.csv", "MIC")
        X, y = data_handler.multi_split_data(prices, self.steps, self.out_steps)

        split_ratio = [9, 1]
        train_X, test_X = data_handler.split_train_test_data(X, split_ratio)
        train_X = train_X.reshape((-1, self.steps, 1))
        test_X = test_X.reshape((-1, self.steps, 1))
        train_y, test_y = data_handler.split_train_test_data(y, split_ratio)
        train_y = train_y.reshape((-1, self.out_steps))
        test_y = test_y.reshape((-1, self.out_steps))
        self.train(train_X, train_y, 200, verbose=1)
        self.test(test_X, test_y, verbose=1)
        self.save()


class Multivariate_LSTM(NeuralNetwork.NeuralNetwork):
    def __init__(self, input_shape, filename):
        self.input_shape = input_shape
        self.model = Sequential()
        self.steps = input_shape[0]
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)
     
        self.model.add(LSTM(8, 
                            activation='relu', input_shape=input_shape))
        # self.model.add(LSTM(8,  return_sequences=True, activation='relu'))
        # self.model.add(LSTM(6, activation='relu'))
        self.model.add(Dense(1))
        self.model.compile(optimizer=optimizer, metrics=['accuracy'], loss='mse')
        self.n_features = self.input_shape[1]
        self.filename = filename
    
    def run_all(self):
        train, labels = data_handler.read_data_from_multiple_files()
        # print(train.shape, labels.shape)
        test_X = data_handler.read_all_data("./Data/trial0010.csv")
        test_y = data_handler.read_data2("./Data/trial0010.csv","TAR")
        
      

        test_X = np.reshape(test_X,(-1, self.steps, 7))
       

        test_y = np.reshape(test_y, (-1, 1))
        # print(test_y)
        self.train(train, labels, epochs =3)
        self.test(test_X, test_y)
        # # yhat = self.model.predict(input, verbose=verbose)
        # # print(y[i], yhat[0][0])
        # # self.test(test_X, test_y, verbose=1)
        self.save()
if __name__ == '__main__':
    
    ## single step vanilla LSTM
    # steps = 9
    # vanilla = Vanilla_LSTM((steps,1),  f"MIC_Predictor_{steps}.8")
    # vanilla.run_all2()

    # multiple step vanilla LSTM
    # in_steps = 12
    # out_steps = 36
    # mul = MultiVanilla_LSTM((in_steps,1), out_steps, f"MIC_MUL_Predictor")
    # mul.run_all()

    # multivariate LSTM
    no_features = 7
    no_steps = 1

    mv = Multivariate_LSTM((no_steps, no_features), f"multivariate_network")
   

    mv.run_all()




    
    
