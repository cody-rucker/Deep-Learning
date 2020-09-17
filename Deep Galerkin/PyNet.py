import numpy as np
import ngsolve as ng

r = ng.x
z = ng.y

Wx = np.genfromtxt('NN_params/Wx.csv', delimiter=',')
Wy = np.genfromtxt('NN_params/Wy.csv', delimiter=',')
Wz = np.genfromtxt('NN_params/Wz.csv', delimiter=',')
b1 = np.genfromtxt('NN_params/b1.csv', delimiter=',')
W2 = np.genfromtxt('NN_params/W2.csv', delimiter=',')
b2 = np.genfromtxt('NN_params/b2.csv', delimiter=',')
W3 = np.genfromtxt('NN_params/W3.csv', delimiter=',')
b3 = np.genfromtxt('NN_params/b3.csv', delimiter=',')

def σ(x):
    return 1 / (1 + np.exp(-x))

# vectorize the logistic function for component-wise application
vσ = np.vectorize(σ)

def u(x, y):
    return W3.dot( vσ(W2.dot( vσ(Wx.dot(x) + Wy.dot(y) + b1)) + b2)) + b3
    
print(u(2.1, 3.4))
