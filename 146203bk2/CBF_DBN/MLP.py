import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def cross_entropy(pred, y):
    return -(y * torch.log(pred + 1e-4)).mean()

def squared_error(pred, y):
    return ((pred - y) ** 2).mean() / 2

class MLP(nn.Module):
    def __init__(self, act_type, opt_type, layers, epochs=20, regression=False, learning_rate=0.01, lmbda=1e-2):
        super(MLP, self).__init__()
        
        act_funcs = {'ReLU': nn.ReLU(), 'Sigmoid': nn.Sigmoid(), 'Tanh': nn.Tanh()}
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.regression = regression
        self.loss = squared_error if regression else cross_entropy
        self.lmbda = lmbda
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
        self.act_func = act_funcs[act_type]
        self.optimizer_type = opt_type
        self.to(self.device)

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.act_func(self.layers[i](x))
        return self.layers[-1](x) if self.regression else F.softmax(self.layers[-1](x), dim=1)

    def fit(self, x, labels):
        x, labels = torch.tensor(x, dtype=torch.float32).to(self.device), torch.tensor(labels, dtype=torch.float32).to(self.device)
        
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate) if self.optimizer_type == 'Adam' else optim.SGD(self.parameters(), lr=self.learning_rate)
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.forward(x)
            loss = self.loss(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}/{self.epochs}, Loss: {loss.item():.4f}")
    
    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.forward(x).cpu().numpy()