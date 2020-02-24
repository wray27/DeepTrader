import numpy as np
import matplotlib.pyplot as plt
import data_handler
from scipy.stats import skew

def time_series_plot():
    d_types = ["TIME", "MID", "MIC", "IMB", "SPR","BID","ASK","TAR"]
    data = {}
    for d in d_types:
        data[d] = np.array([])
        data[d] = data_handler.read_data3("./Data/trial0001.csv", d)

    for i in range(1, len(d_types)):
        plt.plot(data["TIME"], data[d_types[i]], label=d_types[i])
        plt.legend()
        plt.savefig(f"./Diagrams/trial0001-{d_types[i]}")
        plt.close()
    # plt.show()


def hists():
    d_types = ["TIME", "MID", "MIC", "IMB", "SPR", "BID", "ASK", "TAR"]
    data = {}
    for d in d_types:
        data[d] = np.array([])
        data[d] = data_handler.read_data3("./Data/trial0001.csv", d)
        print(d,"skew :",skew(data[d]))
        print(d,"log skew: ",skew(np.sqrt(data[d])))

    for i in range(1, len(d_types)):
        plt.hist( np.log((data[d_types[i]]+10)), bins=20)
        plt.savefig(f"./Diagrams/Histogram-log-trial0001-{d_types[i]}")
        plt.close()
        
def accuracy_plot(actual, preds, baseline):
    time = np.arange(len(actual))
    plt.plot(time, actual,  label="actual", color='blue')
    plt.plot(time, preds,  label="preds", color='red')
    plt.plot(time, baseline,  label="mean", color='green')
    
    plt.legend()
    plt.show()

def main():
    hists()
    time_series_plot()




if __name__ == '__main__':
    main( )
    
