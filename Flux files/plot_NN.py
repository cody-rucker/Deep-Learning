import ngsolve as ng
import numpy as np

W1r = ng.Matrix(3,1)
W1z = ng.Matrix(3,1)
b1 = ng.Vector(3)

W2 = ng.Matrix(1,3)
b2 = ng.Vector(1)

r = ng.x
z = ng.y

def sigmoid(x):
    for i in range(0, len(x)):
        x[i] = 1 / (1 + np.exp(-x[i]))






sigmoid(W1r * r + W1z * z + b1)
