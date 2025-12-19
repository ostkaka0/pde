# File created 2025-12-09 15:19:32 CET
# h - density
# u - electrostatic potential
# u = g at boundary(known)
# u is a weighted average of fundamental solutions with poles on boundary.
# we choose h so that u = g at boundary

import numpy as np
from scipy import linalg
from abc import ABC, abstractmethod
from tqdm import tqdm # For progress bar

# Closed polar parametic curve of the form: self.R(t) * exp(i t)
class PolarCurve(ABC):
  @abstractmethod
  def R(self, t): pass
  
  @abstractmethod
  def RPrim(self, t): pass

  @abstractmethod
  def RBis(self, t): pass

  def r(self, t):
    return c2v(self.R(t) * np.exp(1j*t))

  def rPrim(self, t):
    return c2v((self.RPrim(t) + 1j*self.R(t)) * np.exp(1j*t))

  # nu = normal
  # Calculated by a 90 degree rotation of the tangent
  def nu(self, t):
    z = (-1j*self.RPrim(t) + self.R(t))
    return c2v(z / np.abs(z) * np.exp(1j*t))

  # Returns 1 if coordinate is inside otherwise nan is returned
  def mask(self, x):
    t = np.atan2(x[...,0], x[...,1])
    return np.where(np.vecdot(x, x) <= self.R(t)**2, 1, float('nan')) # Nan is preferable over 0 because we want a white/transparent background when using plt.imshow.


# Turns complex-valued tensor into real tensor of 2d-vectors: complex (...,) to real (..., 2)
def c2v(z):
  return np.stack((z.real, z.imag), axis=-1)
# Turns tensor of 2d-vectors into complex tensor
def v2c(x):
  return x[..., 0] + 1j*x[..., 1]

## Functions not specific to our problem

def calcKernelMat(t, grad_phi, curve):
  T, S = np.meshgrid(t, t)
  X = curve.r(S)
  Y = curve.r(T)
  mat = np.vecdot(grad_phi(Y-X), curve.nu(T))
  diag = (
    1 / (4*np.pi)
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
  h = linalg.solve(np.eye(N)/2 + kernelMat @ np.diag(dsdt * dt), g)

  # Calculate u
  u = np.zeros(X[...,0].shape, dtype=h.dtype)
  for i, t_i in enumerate(tqdm(t, disable=not show_progress)):
    y_i = curve.r(t_i)
    nu_i = curve.nu(t_i)
    kernel_val = np.vecdot(grad_phi(y_i - X), curve.nu(t_i))
    u += kernel_val * h[i] * dsdt[i]
  u *= dt

  return curve.mask(X) * u

def solve_boundary_v(t, t_odd, kernelMat_odd, dsdt_odd, dt, g_odd, curve):
  N = len(t)
  h_odd = linalg.solve(np.eye(N)/2 + kernelMat_odd @ np.diag(dsdt_odd * dt), g_odd)

  x = v2c(curve.r(t))
  y = v2c(curve.r(t_odd))
  v = np.zeros(len(t), dtype=h_odd.dtype)
  for i, t_odd_i in enumerate(t_odd):
    numerator = v2c(curve.nu(t_odd_i))
    denominator = y[i] - x
    v += 1/(2*np.pi) * (numerator / denominator).imag * h_odd[i] * dsdt_odd[i] * dt
  return v

def solve_u_better(M, t, g, v, x_bounds, curve):
  N = len(t)
  f = g + 1j*v

  y = v2c(curve.r(t))
  dydt = v2c(curve.rPrim(t))

  z1 = np.linspace(x_bounds[0], x_bounds[1], M, dtype=t.dtype)
  z2 = np.linspace(x_bounds[2], x_bounds[3], M, dtype=t.dtype)
  Z1, Z2 = np.meshgrid(z1, z2)
  Z = Z1 + 1j*Z2
  
  numerator = np.zeros((M, M), dtype=Z.dtype)
  denominator = np.zeros((M, M), dtype=Z.dtype)
  for i, t_i in enumerate(tqdm(t)):
    numerator   += (f[i] / (y[i]-Z)) * dydt[i] * 2*np.pi/N
    denominator += (1    / (y[i]-Z)) * dydt[i] * 2*np.pi/N
  u = (numerator / denominator).real
  return curve.mask(c2v(Z)) * u
