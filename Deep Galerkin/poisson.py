import ngsolve as ng
import numpy as np
from netgen.geom2d import unit_square
from ngsolve import CoefficientFunction as CF
#ng.ngsglobals.msg_level = 1

def poisson_solve():
    fes = H1(mesh, order=3, dirichlet="left|right|bottom|top")

    u = fes.TrialFunction()
    v = fes.TestFunction()

    f = LinearForm(fes)
    f += -((4*x**2 - 2) *ng.exp(-x**2) + (4*y**2-2) * ng.exp(-y**2))*v*dx

    a = BilinearForm(fes)
    a += grad(u)*grad(v)*dx

    a.Assemble()
    f.Assemble()

    uₕ = ng.GridFunction(fes)
    uₕ.Set(uexact, ng.BND)

    α = f.vec.CreateVector()
    α.data = f.vec - a.mat * uₕ.vec
    uₕ.vec.data += a.mat.Inverse(freedofs=fes.FreeDofs()) * α

    return uₕ



# read in NNet weights and biases
Wx = np.genfromtxt('NN_params/Wx.csv', delimiter=',')
Wy = np.genfromtxt('NN_params/Wy.csv', delimiter=',')
Wz = np.genfromtxt('NN_params/Wz.csv', delimiter=',')
b1 = np.genfromtxt('NN_params/b1.csv', delimiter=',')
W2 = np.genfromtxt('NN_params/W2.csv', delimiter=',')
b2 = np.genfromtxt('NN_params/b2.csv', delimiter=',')
W3 = np.genfromtxt('NN_params/W3.csv', delimiter=',')
b3 = np.genfromtxt('NN_params/b3.csv', delimiter=',')

# manufactured solution
uexact = ng.exp(-ng.x**2) + ng.exp(-ng.y**2)

# logistic activation function
def σ(x):
    return 1 / (1 + ng.exp(-x))

# vectorize the logistic function for component-wise application
vσ = np.vectorize(σ)

# NNet coefficient function
u_net = W3.dot( vσ(W2.dot( vσ(Wx.dot(ng.x) + Wy.dot(ng.y) + b1)) + b2)) + b3

# unit square domain
#mesh = Mesh(unit_square.GenerateMesh(maxh=0.2))
ngmesh = unit_square.GenerateMesh(maxh=0.2)
ngmesh.Refine()
#ngmesh.Refine()

mesh = ng.Mesh(ngmesh)
mesh.ngmesh.SetGeometry(unit_square)
ng.Draw(mesh)

uₕ = poisson_solve()
ΔFEM = uₕ - uexact
ΔNET = u_net - uexact

FEM_error = ng.sqrt(ng.Integrate(ng.InnerProduct(ΔFEM, ΔFEM), mesh))
NET_error = ng.sqrt(ng.Integrate(ng.InnerProduct(ΔNET, ΔNET), mesh))
print('FEM error = ', FEM_error)
print('NET_error = ', NET_error)


ng.Draw(uₕ)
ng.Draw(u_net, mesh, "u_net")
