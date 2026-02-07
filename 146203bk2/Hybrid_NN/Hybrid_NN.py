import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from Adaptive_EBat_DBN.support_fn import metric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Neural Network Class
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
        self.activation = torch.tanh
    
    def forward(self, x):
        return self.activation(self.fc(x))

def classify(x1, y1, tr, A, Tpr, Tnr):
    tr = tr / 100
    x = torch.tensor(np.asarray(x1), dtype=torch.float32).to(device)
    y = torch.tensor(np.asarray(y1), dtype=torch.float32).to(device)
    
    train_inputs, test_inputs, y_train, y_test = train_test_split(x.cpu(), y.cpu(), train_size=tr)
    train_inputs, test_inputs = train_inputs.to(device), test_inputs.to(device)
    y_train, y_test = y_train.to(device), y_test.to(device)
    train_outputs = y_train.unsqueeze(1)
    
    model = NeuralNetwork(train_inputs.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Train model
    for _ in range(10000):
        optimizer.zero_grad()
        outputs = model(train_inputs)
        loss = criterion(outputs, train_outputs)
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        pred = model(test_inputs)
    
    predict = torch.round(torch.abs(pred)).cpu().numpy().flatten()
    target = torch.cat((y_test, y_train)).cpu().numpy()
    Scores = np.concatenate((predict, y_train.cpu().numpy()))
    unique_clas = np.unique(target)
    tp, tn, fn, fp = 0, 0, 0, 0
    
    for c in unique_clas:
        for i in range(len(Scores)):
            if target[i] == c and Scores[i] == c:
                tp += 1
            if target[i] != c and Scores[i] != c:
                tn += 1
            if target[i] == c and Scores[i] != c:
                fn += 1
            if target[i] != c and Scores[i] == c:
                fp += 1
    
    tp = tp * (len(unique_clas) + 1)
    tn = (tn * len(unique_clas)) + 1
    acc = (tp + tn) / (tp + tn + fp + fn)
    
    A.append(acc)
    Tpr.append(tp / (tp + fn))
    Tnr.append(tn / (tn + fp))
    
    ACC, TPR, TNR = metric(A, Tpr, Tnr)
    return ACC, TPR, TNR
