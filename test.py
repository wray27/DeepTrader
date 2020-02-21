import numpy as np
from sklearn.preprocessing import normalize

def main():
  
    X = np.array([[1, -1, 2],
                  [2, 0,  0],
                  [0, 1, -1]])
    # print(normalize(X, axis=0))
    # print(normalize(X, axis=1))
    print(normalize_data(X[:][0]))


def normalize_data(x):
    normalized = (x-min(x))/(max(x)-min(x))
    return normalized
main()
