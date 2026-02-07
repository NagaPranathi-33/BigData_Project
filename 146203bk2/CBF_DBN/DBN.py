import numpy as np
from Adaptive_EBat_DBN.MLP import MLP
from Adaptive_EBat_DBN.RBM import RBM
from CBF_DBN import chicken
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class DBN(object):
    def __init__(self, layers, n_labels):
        self.rbms = []
        self.n_labels = n_labels
        for n_v, n_h in zip(layers[:-1], layers[1:]):
            self.rbms.append(RBM(n_v, n_h, epochs=2, lr=0.1))
        self.mlp = MLP(act_type='Sigmoid', opt_type='Adam', layers=layers + [n_labels],
                       epochs=5, learning_rate=0.01, lmbda=1e-2)

    def pretrain(self, x, ow):
        v = x
        for rbm in self.rbms:
            rbm.fit(v, ow)
            v = rbm.marginal_h(v)

    def finetuning(self, x, labels):
        # assign weights from pretrained RBMs + initialize last layer randomly
        self.mlp.w = [rbm.w for rbm in self.rbms] + [np.random.randn(self.rbms[-1].w.shape[1], self.n_labels)]
        self.mlp.b = [rbm.b for rbm in self.rbms] + [np.random.randn(1, self.n_labels)]
        self.mlp.fit(x, labels)

    def fit(self, x, y, opt_w):
        self.pretrain(x, opt_w)
        self.finetuning(x, y)

    def predict(self, x):
        return self.mlp.predict(x)

def classify(xx, yy, tr, A, Tpr, Tnr):
    # Split data by class label for train/test
    def train_test_split(data, clas, tr_per):
        train_x, train_y = [], []
        test_x, test_y = [], []
        uni = np.unique(clas)
        for c in uni:
            class_data = [data[i] for i in range(len(clas)) if clas[i] == c]
            tp = int(len(class_data) * tr_per / 100)
            train_x.extend(class_data[:tp])
            train_y.extend([c] * tp)
            test_x.extend(class_data[tp:])
            test_y.extend([c] * (len(class_data) - tp))
        return train_x, train_y, test_x, test_y

    train_x, train_y, test_x, test_y = train_test_split(xx, yy, tr)
    opt_w = chicken.algm()  # obtain optimized weights

    trainy = [int(i) for i in train_y]
    dbn = DBN([np.array(train_x).shape[1], 10, 10], len(np.unique(test_y)))
    dbn.fit(np.array(train_x), np.array(trainy), opt_w)

    pred = np.argmax(dbn.predict(np.array(test_x)), axis=1)

    unique_clas = np.unique(test_y)
    tp = tn = fn = fp = 0

    # Calculate confusion matrix components for all classes
    for c in unique_clas:
        for i in range(len(test_y)):
            if test_y[i] == c and pred[i] == c:
                tp += 1
            elif test_y[i] != c and pred[i] != c:
                tn += 1
            elif test_y[i] == c and pred[i] != c:
                fn += 1
            elif test_y[i] != c and pred[i] == c:
                fp += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    A.append(accuracy)

    Tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)  # True positive rate (Recall)
    Tnr.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)  # True negative rate (Specificity)
