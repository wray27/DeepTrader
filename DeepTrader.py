import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.optimizers import Adam
from NeuralNetwork import NeuralNetwork as nn
from Trader import Trader
from Order import Order

def softmax(X):
    expo = np.exp(X)
    expo_sum = np.sum(np.exp(X))
    return expo/expo_sum

def sigmoid(X):
       return 1/(1+np.exp(-X))

class DeepTrader(Trader):
    
    def __init__(self, ttype, tid, balance, time, filename):
        self.ttype = ttype      # what type / strategy this trader is
        self.tid = tid          # trader unique ID code
        self.balance = balance  # money in the bank
        self.blotter = []       # record of trades executed
        # customer orders currently being worked (fixed at 1)
        self.orders = []
        self.n_quotes = 0       # number of quotes live on LOB
        self.willing = 1        # used in ZIP etc
        self.able = 1           # used in ZIP etc
        self.birthtime = time   # used when calculating age of a trader/strategy
        self.profitpertime = 0  # profit per unit time
        self.n_trades = 0       # how many trades has this trader done?
        self.lastquote = None   # record of what its last quote was
        self.filename = filename
        self.model = nn.load_network(self.filename)
        self.model.normalization_values()

    def create_input(self, lob):
        time = lob['time'] 
        bids = lob['bids'] 
        asks = lob['asks'] 
        qid = lob['QID'] 
        tape = lob['tape'] 
        limit = self.orders[0].price
        otype = self.orders[0].otype
        val = 0

        if otype == "Ask": val = 1
       
        mid_price = 0
        micro_price = 0
        imbalances = 0
        spread = 0
        delta_t = 0
        weighted_moving_average = 0

        if len(tape) != 0:

            tape = reversed(tape)
            trades = list(filter(lambda d: d['type'] == "Trade", tape))
            trade_prices = [t['price'] for t in trades]
            delta_t = time - trades[0]['time']

            if (time == trades[0]['time']):
               
                trade_prices = trade_prices[1:]
                if len(trades) == 1:
                    delta_t = trades[0]['time'] - 0
                else:
                    delta_t = trades[0]['time'] - trades[1]['time']

            if len(trade_prices) != 0:
                weights = [(10/9)*(pow(0.9, i)) for i in range(len(trade_prices))]
                weighted_moving_average = sum([a * b for a, b in zip(trade_prices, weights)]) / len(trade_prices)
        else:
            delta_t = time

        if (bids['best'] == None):
            x = 0
        else:
            x = bids['best']
        
        if (asks['best'] == None):
            y = 0
        else:
            y = asks['best']
        
        n_x = bids['n']
        n_y = asks['n']

        spread = abs(y - x)
        mid_price = (x + y) / 2
        if (n_x + n_y != 0 ): 
            micro_price = ((n_x * y) + (n_y * x)) / (n_x + n_y)
            imbalances = (n_x - n_y) / (n_x + n_y)

        market_conditions = np.array([time, val, limit, mid_price, micro_price, imbalances, spread, n_x, n_y])

        return market_conditions
   

    def getorder(self, time, countdown, lob):
        qid = lob['QID']
        tape = lob['tape']
        otype = self.orders[0].otype

        if len(self.orders) < 1:
                # no orders: return NULL
                order = None
        else:
            # creating the input for the network
            x = self.create_input(lob)
            normalized_input = (x-self.model.min_vals) / (self.model.max_vals-self.model.min_vals)

            # dealing witht the networks output
            normalized_output = self.model.predict(x)[0][0]
            denormalized_output = (normalized_output) * ((self.model.max_vals - self.model.max_vals) + self.model.min_vals)
            model_price = int(round(denormalized_output, 0))
           
            order = Order(self.tid, otype, model_price ,
                            self.orders[0].qty, time, qid)
            self.lastquote = order
        return order
    


