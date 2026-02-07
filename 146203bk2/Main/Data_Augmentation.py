import random, numpy as np
import itertools

def data_aug(input_data, clas,dts): # oversampling
    input_data = np.array(input_data)
    clas = np.array(clas)

    # keep augmentation bounded to avoid very large training sets and long runtimes
    base_count = max(len(clas), len(input_data))
    if dts == 'Adult':
        total_instance = min(max(base_count * 2, 2000), 4000)
    else:
        total_instance = min(max(base_count * 2, 500), 1000)
    # input_data = list(itertools.chain(*input_data))
    # clas = list(itertools.chain(*clas))
    def augment(data,cls, ins_total):
        da,lab=[], []
        c_min, c_max = [], []   # column min & max
        l_min , l_max = [],[]
        for i in range(len(data[0])):   # for each feature
            col = data[:, i]
            c_min.append(np.min(col))   # add min value
            c_max.append(np.max(col))   # add max value
            l_min.append(0)
            l_max.append(1)

        for i in range(ins_total): # data augmentation for class label
            if i>=len(cls):
                lab.append(random.randint(l_min[0], l_max[0]))
            else:
                lab.append(cls[i])

        for i in range(ins_total): # data augmentation for feature
            if i < len(data):
                da.append(data[i])
            else:
                tmp = []
                for j in range(len(data[0])):
                    # for extra instance generate random b/w min & max of that feature
                    tmp.append(random.uniform(c_min[j], c_max[j]))
                da.append(np.array(tmp))
        return da,lab
    Data,clas = augment(input_data,clas, total_instance)
    return Data, clas
