import numpy as np
import sympy
from sympy.abc import x, y
from matplotlib import pyplot as plt
import matplotlib.patches as patches

def cylinder_stream_function(U = 1, R = 1):
  r = sympy.sqrt(x ** 2 + y ** 2)
  theta = sympy.atan2(y, x)
  return U * (r - R ** 2 / r) * sympy.sin(theta)

def velocity_field(psi):
  u = sympy.lambdify((x, y), psi.diff(y), 'numpy')
  v = sympy.lambdify((x, y), -psi.diff(x), 'numpy')
  return u, v

psi = cylinder_stream_function()
U_func, V_func = velocity_field(psi)

xmin, xmax, ymin, ymax = -6, 6, -6, 6
Y, X = np.ogrid[ymin:ymax:128j, xmin:xmax:128j]
U, V = U_func(X, Y), V_func(X, Y)

M = (X ** 2 + Y ** 2) < 1.
U = np.ma.masked_array(U, mask = M)
V = np.ma.masked_array(V, mask = M)
shape = patches.Circle((0, 0), radius = 1., lw = 2., fc = 'w', ec = 'k', zorder = 0)
plt.gca().add_patch(shape)

plt.streamplot(X, Y, U, V, color = U ** 2 + V ** 2, cmap = plt.cm.binary)
plt.show()