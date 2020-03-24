import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import model_from_json


class NeuralNetwork():
    
    def __init__(self):
        pass
    
    def train(self, X, y, epochs, verbose=1):
        self.model.fit(X, y, epochs=epochs, verbose=verbose)
    
    def test(self, X, y, verbose=1):
        for i in range(len(X)):
            input = X[i].reshape((1,self.steps, self.input_shape[1]))
            yhat = self.model.predict(input, verbose=verbose)
            print(y[i], yhat[0][0])
    
    def save(self):

        # serialize model to JSON
        model_json = self.model.to_json()
        with open("./Models/" + self.filename, "w") as json_file:
            json_file.write(model_json)
        
        # serialize weights to HDF5
        self.model.save_weights("./Models/" + self.filename + ".h5")
        print("Saved model to disk")

    @staticmethod
    def load_network(filename):
        
        # load json and create model
        json_file = open("./Models/" + filename, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        
        # load weights into new model
        loaded_model.load_weights("./Models/" + filename + ".h5")
        # print("Loaded model from disk.")
        
        return loaded_model



