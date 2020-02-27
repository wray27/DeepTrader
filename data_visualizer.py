import numpy as np
import matplotlib.pyplot as plt
import data_handler
import seaborn as sns
from scipy.stats import skew

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
        
def accuracy_plot(actual, preds, baseline):
    time = np.arange(len(actual))
    plt.plot(time, actual,  label="actual", color='blue')
    plt.plot(time, preds,  label="preds", color='red')
    plt.plot(time, baseline,  label="mean", color='green')
    
    plt.legend()
    plt.show()

def relationships(filename):
    d_types = ["TIME", "MID", "MIC", "IMB", "SPR", "BID", "ASK", "TAR"]
    data = {}
    # print(filename)
    for d in d_types:
        data[d] = np.array([])
        data[d] = data_handler.read_data2('./Data/' + filename + ".csv", d)
    corr = np.corrcoef([data[d] for d in data])
    sns.heatmap(corr)
    plt.show()

def main():
    # for i in range(1,11):
    #     filename = 'trial%04d' % i
    #     hists(filename)
    relationships("trial0009")
    



if __name__ == '__main__':
    main( )
    
