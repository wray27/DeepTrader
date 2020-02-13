import numpy as np
import matplotlib.pyplot as plt

def time_series_plot():
    d_types = ["TIME", "MID", "MIC", "IMB", "SPR"]
    data = {}
    for d in d_types:
        data[d] = np.array([])
        data[d] = read_data("lob_data.csv", d)

    # for i in range(1, len(d_types)):
    #     plt.plot(data["TIME"], data[d_types[i]], label=d_types[i])
    plt.plot(data["TIME"], data[d_types[4]], label=d_types[4])

    plt.legend()
    plt.show()


def accuracy_plot(actual, preds, baseline):
    time = np.arange(len(actual))
    plt.plot(time, actual,  label="actual", color='blue')
    plt.plot(time, preds,  label="preds", color='red')
    plt.plot(time, baseline,  label="mean", color='green')
    
    plt.legend()
    plt.show()






if __name__ == '__main__':
    pass
    
