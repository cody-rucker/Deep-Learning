import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

N, D_in, H, D_out = 16, 2, 25, 1
max = 1
min = 0

# advection velocity
c =2* torch.ones(N)
one = torch.ones(N)
# initial condition
def u0(x):
    return torch.sin(2*np.pi*x[:,0])

# boundary condition at x = 1
def g1(t):
    return torch.sin(2*np.pi*(torch.ones(len(t)) + c*t[:,1]))

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
        return y_pred[:,0]

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

def F(z):
    #gradx = torch.zeros_like(z[:,0])
    #gradx = gradx.unsqueeze(1)
    #gradt = torch.zeros_like(z[:,0])
    #gradt = gradt.unsqueeze(1)

    gradz = torch.zeros_like(z[:,0])
    #gradz = gradz.unsqueeze(1)

    for i in range(len(z)):
        #qx = torch.autograd.grad(model(z)[i], x0, retain_graph=True, create_graph=True)[0]
        #qt = torch.autograd.grad(model(z)[i], t0, retain_graph=True, create_graph=True)[0]
        qz = torch.autograd.grad(model(z)[i], z, retain_graph=True, create_graph=True)[0]
        qz = qz[:, 0] - qz[:, 1]
        gradz = gradz + qz
        #gradx = gradx + qx
        #gradt = gradt + qt

    return gradz

def cost(z0, z1, z2):
    # impose structure of PDE
    s0 = torch.sum(torch.square(torch.abs( F(z0))))
    # impose initial condition
    s1 = torch.sum(torch.square(torch.abs( model(z1) - u0(z1))))

    #impose boundary condition
    s2 = torch.sum(torch.square(torch.abs( model(z2) - g1(z2))))
    return s0 + s1 + s2

# N is batch size; D_in is input dimension;
# H is hidden dimension(i.e number of neurnas)
#; D_out is output dimension.
#N, D_in, H, D_out = 64, 2, 10, 1
#z = torch.arange(0, 1, 0.03125)
#max = 1
#min = 0
#x = torch.randn(N, D_in)
#x = (max-min)*torch.rand((N, 1)) + min
# = torch.sin(torch.randn(N, D_out))
#y = torch.sin(2*np.pi*x)

#model = TwoLayerNet(D_in, H, D_out)

#criterion = torch.nn.MSELoss(reduction='sum')
#criterion = cost
#x1 = (max-min)*torch.rand((N, 1), requires_grad=True) + min
#t1 = torch.rand((N, 1), requires_grad=True)
#z1 = torch.cat([x1, t1], dim=1)


#print(torch.sum(torch.square(torch.abs( F(z1)))))
#print(z1)
#q = torch.autograd.grad(model(z1)[0], z1, create_graph=True)
#print(q)
#print(x1.grad)
#loss = criterion(z1)

#print(model(z1))
#print(u0(z1))
model = TwoLayerNet(D_in, H, D_out)
criterion = cost
#print(F(z1))
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

for t in range(12000):

    # random domain point for training PINN
    x0 = (max-min)*torch.rand((N, 1), requires_grad=True) + min
    t0 = (max-min)*torch.rand((N, 1), requires_grad=True) + min
    z0 = torch.cat([x0,t0], dim=1)

    # random initial point
    x1 = (max-min)*torch.rand((N, 1)) + min
    t1 = 0*torch.rand((N, 1))
    z1 = torch.cat([x1, t1], dim=1)

    # random boundary point
    x2 = torch.ones(N, 1)
    t2 = (max-min)*torch.rand((N, 1)) + min
    z2 = torch.cat([x2, t2], dim=1)


    # Compute and print loss
    #loss = criterion(y_pred, y)
    loss = criterion(z0, z1, z2)
    if t % 100 == 99:
        print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
#out = model(x)
t = torch.arange(0,1,0.05)
for i in range(len(t)):
    z = torch.ones(N, 1)
    Z = torch.arange(0, 1, 1/N)
    v = t[i]*torch.ones(N, 1)
    for i in range(N):
        z[i] = Z[i]
    V = torch.cat([z, v], dim=1)
    out = model(V)
    print(out)

#out = model(z)
#exact = torch.sin(2*np.pi*z)

    plt.plot(z.detach().numpy(), model(V).detach().numpy(), label='network')
    #plt.plot(z.detach().numpy(), exact.detach().numpy(), label='exact')
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
