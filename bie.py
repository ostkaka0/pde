# © 2025 John Emanuelsson
# File created 2025-12-09 15:19:32 CET

# h - density
# u - electrostatic potential
# u = g at boundary(known)
# u is a weighted average of fundamental solutions with poles on boundary.
# we choose h so that u = g at boundary

import numpy as np
from scipy import linalg
from abc import ABC, abstractmethod
from tqdm import tqdm

# Closed polar parametic curve of the form: self.R(t) * exp(i t)
class PolarCurve(ABC):
  @abstractmethod
  def R(self, t): pass
  
  @abstractmethod
  def RPrim(self, t): pass

  @abstractmethod
  def RBis(self, t): pass

  def r_complex(self, t):
    return self.R(t) * exp(1j * t)

  def rPrim_complex(self, t):
    return (self.RPrim(t) + 1j*self.R(t)) * exp(1j * t)

  def rBis_complex(self, t):
    return (self.RBis(t) + 2j*self.RPrim(t) - self.R(t)) * exp(1j * t)

  def r(self, t):
    return c2v(self.r_complex(t))

  def rPrim(self, t):
    return c2v(self.rPrim_complex(t))

  def rBis(self, t):
    return c2v(self.rBis_complex(t))
  
  # Calculated by a 90 degree rotation of the tangent
  def nu_complex(self, t): # nu = normal
    return -1j * self.rPrim_complex(t) / np.abs(self.rPrim_complex(t))

  def nu(self, t): # nu = normal
    return c2v(self.nu_complex(t))

  # Returns 1 if coordinate is inside otherwise nan is returned
  def mask(self, x):
    return 1
    t = np.atan2(x[...,0], x[...,1])
    return np.where(vecdot(x, x) <= self.R(t)**2, 1, float('nan')) # Nan is preferable over 0 because we want a white/transparent background when using plt.imshow.
  

## Aliases
## Note: some aliases overrides built-in math-function
pi = np.pi
abs = np.abs
pow = np.pow
exp = np.exp
cos = np.cos
sin = np.sin
sqrt = np.sqrt
real = np.real
imag = np.imag
conj = np.conj
log = np.log
log10 = np.log10
eye = np.eye
diag = np.diag
vecdot = np.linalg.vecdot



# Turns complex-valued tensor into real tensor of 2d-vectors: complex (...,) to real (..., 2)
def c2v(z):
  return np.stack((z.real, z.imag), axis=-1)

## Functions not specific to our problem

def calcKernelMat(t, grad_phi, curve):
  T, S = np.meshgrid(t, t)
  X = curve.r(S)
  Y = curve.r(T)
  mat = vecdot(grad_phi(Y-X), curve.nu(T))
  diag = (
    1 / (4*pi)
    * (-curve.RBis(t) * curve.R(t)  + 2*curve.RPrim(t)**2  + curve.R(t)**2)
    / pow(curve.RPrim(t)**2 + curve.R(t)**2, 3/2)
  )
  # Insert the diagonals
  idcs = np.arange(len(t))
  mat[idcs, idcs] = diag

  return mat


## BIE-algorithms
def solve_u(kernelMat, X, t, dsdt, dt, g, grad_phi, curve, show_progress=True):
  N = len(t)
  h = linalg.solve(eye(N)/2 + kernelMat @ diag(dsdt * dt), g)

  # Calculate u
  u = np.zeros(X[...,0].shape, dtype=h.dtype)
  for i, t_i in enumerate(tqdm(t, disable=not show_progress)):
    y_i = curve.r(t_i)
    nu_i = curve.nu(t_i)
    kernel_val = vecdot(grad_phi(y_i - X), curve.nu(t_i))
    u += kernel_val * h[i] * dsdt[i]
  u *= dt

  return curve.mask(X) * u

def solve_boundary_v(t, t_odd, kernelMat_odd, dsdt_odd, dt, g_odd, curve):
  N = len(t)
  h_odd = linalg.solve(eye(N)/2 + kernelMat_odd @ diag(dsdt_odd * dt), g_odd)

  x = curve.r_complex(t)
  y = curve.r_complex(t_odd)
  v = np.zeros(len(t), dtype=h_odd.dtype)
  for i, t_odd_i in enumerate(t_odd):
    numerator = curve.nu_complex(t_odd_i)
    denominator = y[i] - x
    v += 1/(2*pi) * imag(numerator / denominator) * h_odd[i] * dsdt_odd[i] * dt
  return v

def solve_u_better(M, t, g, v, x_bounds, curve):
  N = len(t)
  f = g + 1j*v

  y = curve.r_complex(t)
  dydt = curve.rPrim_complex(t)

  z1 = np.linspace(x_bounds[0], x_bounds[1], M, dtype=t.dtype)
  z2 = np.linspace(x_bounds[2], x_bounds[3], M, dtype=t.dtype)
  Z1, Z2 = np.meshgrid(z1, z2)
  Z = Z1 + 1j*Z2
  # Z = np.array(np.meshgrid(z1, z2))
  
  numerator = np.zeros((M, M), dtype=Z.dtype)
  denominator = np.zeros((M, M), dtype=Z.dtype)
  for i, t_i in enumerate(tqdm(t)):
    numerator   += (f[i] / (y[i]-Z)) * dydt[i] * 2*pi/N
    denominator += (1    / (y[i]-Z)) * dydt[i] * 2*pi/N
  u = real(numerator / denominator)
  return curve.mask(c2v(Z)) * u


