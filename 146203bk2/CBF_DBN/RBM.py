import torch
import torch.nn as nn
import torch.optim as optim

def sigmoid(x):
    return torch.sigmoid(x)

class RBM(nn.Module):
  def __init__(self, n_v, n_h, epochs=2, lr=0.05):
    super(RBM, self).__init__()
    self.n_v = n_v
    self.n_h = n_h
    self.lr = lr
    self.epochs = epochs
    self.batch_size = 16
    self.decay = 1 - 1e-4
        
    # Initialize weights and biases
    self.W = nn.Parameter(torch.randn(n_v, n_h, device="cuda"))
    self.a = nn.Parameter(torch.randn(1, n_v, device="cuda"))
    self.b = nn.Parameter(torch.randn(1, n_h, device="cuda"))
        
    # Optimizer
    self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

  def forward(self, v):
    p_h = sigmoid(torch.matmul(v, self.W) + self.b)
    return p_h

  def sample_h(self, v):
    p_h = self.forward(v)
    return (p_h >= torch.rand_like(p_h)).float()

  def sample_v(self, h):
    p_v = sigmoid(torch.matmul(h, self.W.t()) + self.a)
    return (p_v >= torch.rand_like(p_v)).float()

  def fit(self, data):
    data = torch.tensor(data, dtype=torch.float32, device="cuda")
    train_num = data.shape[0]
    for epoch in range(self.epochs):
      permut = torch.randperm(train_num).split(self.batch_size)
      for batch in permut:
        v0 = data[batch]
        h0 = self.sample_h(v0)
        v1 = self.sample_v(h0)
        h1 = self.sample_h(v1)
                
        # Compute gradients
        grad_w = torch.matmul(v0.t(), h0) - torch.matmul(v1.t(), h1)
        grad_a = torch.sum(v0 - v1, dim=0, keepdim=True)
        grad_b = torch.sum(h0 - h1, dim=0, keepdim=True)
                
        # Update weights and biases
        self.optimizer.zero_grad()
        self.W.grad = -grad_w / self.batch_size
        self.a.grad = -grad_a / self.batch_size
        self.b.grad = -grad_b / self.batch_size
        self.optimizer.step()
                
        # Apply decay
        self.W.data *= self.decay
        self.a.data *= self.decay
        self.b.data *= self.decay
