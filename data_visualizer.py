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


def forecast_plot(time, actual, pred_times, preds):
    plt.plot(time, actual,  label="actual", color='blue')
    plt.plot(pred_times, preds, label="prediction", color='red')

    plt.legend()
    plt.show()






if __name__ == '__main__':
    pass
    
