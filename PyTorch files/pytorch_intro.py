import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

'''
def jacobian(y, x, create_graph=False):
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.
    return torch.stack(jac).reshape(y.shape + x.shape)

def hessian(y, x):
    return jacobian(jacobian(y, x, create_graph=True), x)

def f(x):
    return x * x * torch.arange(4, dtype=torch.float)

def g(x):
    return torch.sin(x)*torch.ones(2)


x = torch.ones(4, requires_grad=True)
#print(jacobian(f(x), x))
#print(hessian(f(x), x))

x = torch.tensor(1.0, requires_grad=True)
print(jacobian(g(x), x))
print(hessian(g(x), x))
'''

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_sig = torch.sigmoid(self.linear1(x))
        y_pred = self.linear2(h_sig)
        return y_pred
        
# N is batch size; D_in is input dimension;
# H is hidden dimension(i.e number of neurnas)
#; D_out is output dimension.
N, D_in, H, D_out = 128, 1, 10, 1
#z = torch.arange(0, 1, 0.03125)
max = 1
min = 0
#x = torch.randn(N, D_in)
x = (max-min)*torch.rand((N, 1)) + min
# = torch.sin(torch.randn(N, D_out))
y = torch.sin(2*np.pi*x)

model = TwoLayerNet(D_in, H, D_out)
print(model.forward(x[1]))
#print(model(x))
#print(y)

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
for t in range(2000):
    x = (max-min)*torch.rand((N, 1)) + min
    y = torch.sin(2*np.pi*x)
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
out = model(x)

z = torch.ones(N, 1)
Z = torch.arange(0, 1, 1/N)
for i in range(N):
    z[i] = Z[i]
out = model(z)
exact = torch.sin(2*np.pi*z)

plt.plot(z.detach().numpy(), out.detach().numpy(), label='network')
plt.plot(z.detach().numpy(), exact.detach().numpy(), label='exact')
plt.legend()
plt.show()
#
#for i in range(len(z)):
#    Z[i] = model.forward(z[i])
#z = torch.arange(0, 1, 0.03125)
#Z = model(z[1])
#y = x.grad

#z = torch.tensor(1.0)
#print(z.double())
