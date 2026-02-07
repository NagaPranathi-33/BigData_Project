import numpy as np
from scipy.spatial import distance

def by_find_kulczynski(data, clas):

    data = np.transpose(data)   # transpose for attribute selection

    # kulczynski's polynomial
    h_value = []
    for i in range(len(data)):
        # print(i)
        h_value.append(distance.kulczynski1(data[i], clas))

    # Sort attribute by min
    H = h_value.copy()
    H.sort()

    # sorted index
    sort_index = []
    for i in range(len(H)):
        ind = h_value.index(H[i])     # index of sorted data
        sort_index.append(ind)
        h_value[ind] = np.inf   # set by max value to avoid repeated value confusion

    # sort Attributes by sorted index
    sorted_attr = []
    for i in range(len(sort_index)):
        sorted_attr.append(data[sort_index[i]])   # add attribute according to the sorted index
    return np.transpose(sorted_attr)    # transpose to get its original form
