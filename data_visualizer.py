import numpy as np
import matplotlib.pyplot as plt
import data_handler
import seaborn as sns
from scipy.stats import skew
import matplotlib as mpl

def time_series_plot(filename):
    d_types = ["TIME", "MID", "MIC", "IMB", "SPR","BID","ASK","TAR"]
    data = {}
    for d in d_types:
        data[d] = np.array([])
        data[d] = data_handler.read_data3("./Data/" + filename + ".csv", d)

    for i in range(1, len(d_types)):
        plt.plot(data["TIME"], data[d_types[i]], label=d_types[i])
        plt.legend()
        plt.savefig(f"./Diagrams/{filename}-{d_types[i]}")
        plt.close()
    


def hists(filename):
    d_types = ["TIME", "MID", "MIC", "IMB", "SPR", "BID", "ASK", "TAR"]
    data = {}
    print(filename)
    for d in d_types:
        data[d] = np.array([])
        data[d] = data_handler.read_data3('./Data/' + filename + ".csv", d)
        print(d,"skew :",skew(data[d]))
        print(d,"log skew: ",skew(np.sqrt(data[d])))

    for i in range(1, len(d_types)):
        plt.hist( data[d_types[i]], bins=20)
        plt.savefig(f"./Diagrams/Histogram-{filename}-{d_types[i]}")
        plt.close()
        
def accuracy_plot(actual, preds, baseline=[]):
    time = np.arange(len(actual))
    mpl.style.use('seaborn')
    plt.plot(time, actual,  label="actual", color='red')
    plt.plot(time, preds,  label="preds", color='green')

    plt.title("Predicting the Micro Price in Bristol Stock Exchage")

    # plt.plot(time, baseline,  label="mean", color='green')
    plt.xlabel("Prediction Number")
    plt.ylabel("Micro Price")
    plt.legend()
    plt.show()

def relationships():
    n = 3
    file = f"./Data/Training/trial{(n):04}.csv"
    data = data_handler.read_all_data(file)
   
    corr = np.corrcoef([data[:][d] for d in range(data.shape[1])])
    sns.heatmap(corr)
    plt.show()

def profit_time(number):

    market_data, trader_data = data_handler.collect_results(number)
    # mpl.style.use('seaborn')

    for t in trader_data.keys():
        plt.plot(market_data["TIME"], trader_data[t]["PPT"],  label=t)
    
    plt.title("Profit per Trader over Time in BSE")
    plt.grid()
    plt.xlabel("Time (s)")
    plt.ylabel("Profit per Trader")
    plt.legend()
    plt.show()


def main():
   
    relationships()
    # for i in range(1,11):
    #     profit_time(i)



if __name__ == '__main__':
    main( )
    
