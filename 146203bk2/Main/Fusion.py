import numpy as np
import Sort
import DNN

def feature_fusion(data,clas):
    data = np.array(data)
    clas = np.array(clas)
    # arrange features by jaccard distance
    f = Sort.by_find_kulczynski(data,clas)

    T = len(f)
    F_new, l = [], round(T)
    S = (len(data[0]))
    # train_x, train_y = preprocess(data, clas)
    beta = DNN.classify(data, clas)  # beta value prediction by Deep residual Network

    for m in range(len(f)):  # data size
        FF = []
        for n in range(S):  # n_features to be fused
            summ, i, p = 0, n, 1
            while p <= l:  # n attributed to fused as 1
                summ += ((beta[m] / p)*f[m][i])
                if (i + S) < 1: i = i + S
                else: p += l  # last set
                p += 1
            FF.append(summ)
        F_new.append(FF)
        # F_new.append(feat)
    return F_new