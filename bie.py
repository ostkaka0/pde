#!/usr/bin/python
# File created 2025-12-09 15:19:32 CET

# h - density
# u - electrostatic potential
# u = g at boundary(known)
# u is a weighted average of fundamental solutions with poles on boundary.
# we choose h so that u = g at boundary

 
## Parse arguments before anything else
import argparse
parser = argparse.ArgumentParser();
parser.add_argument(
  "--pytorch",
  action="store_true",
  default=False
)
parser.add_argument(
  "-N",
  type=int,
  default=200
)
parser.add_argument(
  "-M",
  type=int,
  default=200
)
parser.add_argument(
  "--float",
  type=int,
  default=64
)
parser.add_argument(
  "-q",
  type=int,
  default=0
)
parser.add_argument(
  "-p",
  type=float,
  default=3
)
parser.add_argument(
  "-k",
  type=float,
  default=1
)
parser.add_argument(
  "--cheat",
  action='store_true'
)
parser.add_argument(
  "--helm",
  action='store_true'
)
args = parser.parse_args()
N = args.N
M = args.M
p = args.p
k = args.k

## Imports
np = None
if not args.pytorch:
  import numpy as np
  from scipy import linalg
else:
  import torch
  from torch import linalg
  np = torch
import matplotlib.pyplot as plt
from tqdm import tqdm # Progress bar
from scipy import special
from abc import ABC, abstractmethod

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
    t = np.atan2(x[...,0], x[...,1])
    return np.where(vecdot(x, x) <= curve.R(t)**2, 1, float('nan')) # Nan is preferable over 0 because we want a white/transparent background when using plt.imshow.
  
class OurCurve(PolarCurve):
  def R(self, t):
    return (3 + cos(4*t + pi))
  def RPrim(self, t):
    return (-4 * sin(4*t + pi))
  def RBis(self, t):
    return (-16 * cos(4*t + pi))


## Set correct dtype
dtype = {
  32: np.float32,
  64: np.float64,
  128: np.float128 if not args.pytorch else None
}[args.float]
complex_dtype = {
  32: np.complex64,
  64: np.complex128,
  128: np.complex256 if not args.pytorch else None
}[args.float]
print("dtypes:", dtype, ",", complex_dtype)

if args.pytorch:
  print("Using pytorch")
  print("CPU threads used by pytorch:", torch.get_num_threads())

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

## Helper functions
def is_complex(x):
  return torch.is_complex(x) if args.pytorch else np.iscomplexobj(x)
def vecnorm(x):
  return sqrt(vecdot(x, x))

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

  z1 = np.linspace(x_bounds[0], x_bounds[1], M, dtype=dtype)
  z2 = np.linspace(x_bounds[2], x_bounds[3], M, dtype=dtype)
  Z1, Z2 = np.meshgrid(z1, z2)
  Z = Z1 + 1j*Z2
  # Z = np.array(np.meshgrid(z1, z2))
  
  numerator = np.zeros((M, M), dtype=complex_dtype)
  denominator = np.zeros((M, M), dtype=complex_dtype)
  for i, t_i in enumerate(tqdm(t)):
    numerator   += (f[i] / (y[i]-Z)) * dydt[i] * 2*pi/N
    denominator += (1    / (y[i]-Z)) * dydt[i] * 2*pi/N
  u = real(numerator / denominator)
  return curve.mask(c2v(Z)) * u


## Plotting functions
# Plot A, B, and the log-abs error of A assuming B is the correct solution.
def plot_mat_comparison_and_show(A, B, extent=None, vmin=None, vmax=None):
  log_abs_err = log10(abs(A - B))
  fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)

  im0 = ax[0].imshow(real(A.T), origin = 'lower', vmin=vmin, vmax=vmax, extent=extent)
  im1 = ax[1].imshow(real(B.T), origin = 'lower', vmin=vmin, vmax=vmax, extent=extent)
  im2 = ax[2].imshow(log_abs_err.T, origin = 'lower', cmap='turbo', extent=extent)
  for a in ax:
    a.set_aspect('equal', adjustable='box')
    a.set_axis_off()
    # a.set_xlabel("x1")
    # a.set_ylabel("x2")
  plt.colorbar(im0)
  plt.colorbar(im1)
  plt.colorbar(im2)
  ax[0].set_title("approximation")
  ax[1].set_title("Correct solution")
  ax[2].set_title("Log-abs error")
  plt.tight_layout()
  plt.show()
# def plot_mat_and_show(mat, extent=None, vmin=None, vmax=None):
#   plt.imshow(real(mat.T), origin = 'lower', cmap='CMRmap_r', vmin=vmin, vmax=vmax, extent=extent)
#   plt.axis('equal')
#   plt.colorbar()
#   plt.show()

# Plot kernel. If complex, we plot both real and imaginary.
def plot_kernel_and_show(mat, extent=None, title=None):
  fig, ax = plt.subplots(1, 2 if is_complex(mat) else 1, squeeze = False)
  ax = ax[0]

  ax[0].set_title(title + ("(real)" if is_complex(mat) else ""))
  ax[0].set_xlabel("s")
  ax[0].set_ylabel("t")
  im1 = ax[0].imshow(real(mat.T), origin = 'lower', cmap='turbo', extent=extent)
  ax[0].axis('equal')
  fig.colorbar(im1)
  if is_complex(mat):
    ax[1].set_title(title + "(imag)")
    ax[1].set_xlabel("s")
    ax[1].set_ylabel("t")
    im2 = ax[1].imshow(imag(mat.T), origin = 'lower', cmap='turbo', extent=extent)
    ax[1].axis('equal')
    fig.colorbar(im2)
  plt.tight_layout()
  plt.show()
    

## Problem specific functions
def phi_k(x):
  return -1j/4 * special.hankel1(0, k * vecnorm(x))
def phi(x):
  return 1/(2*pi) * log(vecnorm(x))
def grad_phi(x):
  x_dot_x = vecdot(x, x)[..., np.newaxis]
  return 1/(2*pi) * x / x_dot_x
def grad_phi_k(x):
  return +1j/4 * k*x/vecnorm(x)[..., np.newaxis] * special.hankel1(1, k*vecdot(x, x))[..., np.newaxis]
def secret_u(r): # u at coord r
  if args.helm:
    return special.hankel1(0, k * vecnorm(r - p))
  else:
    return exp((r[...,0] + 0.3*r[...,1])/3) * sin((0.3*r[...,0] - r[...,1])/3)
  
def secret_v(r): # u at coord r
  return exp((r[...,0] + 0.3*r[...,1])/3) * cos((0.3*r[...,0] - r[...,1])/3)
# Boundary-values
# def g(t, curve):
#   return secret_u(curve.r(t))

### Excersises
curve = OurCurve()
t = np.linspace(-pi + 2*pi/N, pi, N, dtype=dtype)
# t = np.linspace(-pi, pi, N, dtype=dtype, endpoint=False)
t_odd = t - (t[1]-t[0])/2 # Assumes equal spacing between t-values
dt = 2*pi/N
dsdt = vecnorm(curve.rPrim(t))
dsdt_odd = vecnorm(curve.rPrim(t_odd))
x_bounds = (-4, 4, -4, 4)
t_bounds = (t[0], t[-1], t[0], t[-1])
t_bounds_odd = (t_odd[0], t_odd[-1], t_odd[0], t_odd[-1])
x1 = np.linspace(x_bounds[0], x_bounds[1], M, dtype=dtype)
x2 = np.linspace(x_bounds[2], x_bounds[3], M, dtype=dtype)
X = np.stack(np.meshgrid(x1, x2), axis=-1)
our_grad_phi = grad_phi_k if args.helm else grad_phi

# plt.plot(t, kernel_diag())

## Problem 1:
print("Calculating kernel...")
kernelMat = calcKernelMat(t, our_grad_phi, curve);
kernelMat_odd = calcKernelMat(t_odd, our_grad_phi, curve);
if args.q == 1 or args.q == 0:
  plot_kernel_and_show(kernelMat, t_bounds, "Kernel matrix")
  print("Solving u...")
  g = secret_u(curve.r(t))
  u = solve_u(kernelMat, X, t, dsdt, dt, g, our_grad_phi, curve)
  u_correct = secret_u(X) * curve.mask(X)
  plot_mat_comparison_and_show(u, u_correct, x_bounds, -1, 1)

## Problem 2:
g_odd = secret_u(curve.r(t_odd))
v = solve_boundary_v(t, t_odd, kernelMat_odd, dsdt_odd, dt, g_odd, curve)
v_correct = secret_v(curve.r(t))
print("v", v.shape)
print("v_correct", v_correct.shape)
print("t", t.shape)
print("r t", curve.r(t).shape)

if args.cheat:
  v = v_correct

if args.q == 2 or args.q == 0:
  plt.plot(t, v, label="Approximation")
  plt.plot(t, v_correct, ":", label="Correct")
  plt.title("v at boundary")
  plt.show()
  # Plot log-abs-error
  plt.title("v at boundary - Log-abs error")
  log_abs_err_v = log10(abs(v - v_correct + np.mean(v_correct) - np.mean(v)))
  plt.plot(t_odd, log_abs_err_v)
  plt.show()

## Problem 3
print("Calculating odd kernel...")
kernelMat = calcKernelMat(t_odd, our_grad_phi, curve);
if args.q == 3 or args.q == 0:
  plot_kernel_and_show(kernelMat_odd, t_bounds_odd, "Odd kernel matrix")
  g = secret_u(curve.r(t))
  u = solve_u_better(M, t, g, v, x_bounds, curve)
  u_correct = secret_u(X) * curve.mask(X)
  plot_mat_comparison_and_show(u, u_correct, x_bounds, -3, 3)
  # plot_mat_and_show(u, x_bounds, -3, 3)
  # plot_mat_and_show(u_correct, x_bounds, -3, 3)
  # log_abs_err = log10(abs(u - u_correct))
  # plot_mat_and_show(log_abs_err, x_bounds)

# Problem 4
if args.q == 4 or args.q == 0:
  # Measure the far-field error for different N. We will just measure the error at the point(0, 0)
  x = np.array([0, 0])

  Ns        = np.arange(10, 1000, 10)
  u         = np.zeros(Ns.shape)
  u_correct = np.zeros(Ns.shape)
  err       = np.zeros(Ns.shape)
  for i, N2 in enumerate(tqdm(Ns)): 
    t = np.linspace(-pi, pi, N2, dtype=dtype, endpoint=False)
    dsdt = vecnorm(curve.rPrim(t))
    kernelMat = calcKernelMat(t, our_grad_phi, curve)
    g = secret_u(curve.r(t))
    u[i] = solve_u(kernelMat, x, t, dsdt, dt, g, our_grad_phi, curve, show_progress=False)
    u_correct[i] = secret_u(x) * curve.mask(x)
    err[i] = log10(abs(u[i] - u_correct[i]))

  plt.plot(Ns, u, label="u")
  plt.plot(Ns, u_correct, ":", label="u_correct")
  plt.plot(Ns, err, label="err")
  plt.show()
    

# ## Problem 4:
# if args.q == 4 or args.q == 0:
#   d = np.linspace(0, 1.0, 25)
#   D, T = np.meshgrid(d, t)  
#   X2 = curve.r(T) - D * curve.nu(T)
#   # plot_mat_and_show(X2[1], None, -1, 1)

#   plt.plot(X2[0, :], X2[1, :])
#   plt.plot(X2[0, :, 10], X2[1, :, 10])
#   plt.plot(X2[0, :,-1], X2[1, :, -1])
#   plt.show()

#   u2 = solve_u(kernelMat, X2, t, dsdt, dt)
#   u2_correct = correct_u(X2)
#   plot_mat_comparison_and_show(u2, u2_correct, None, -1, 1)
  
#   # plot_mat_and_show(u2, None, -1, 1)
#   # plot_mat_and_show(u2_correct, None, -1, 1)
#   # log_abs_err = log10(abs(u2 - u2_correct))
#   # plot_mat_and_show(log_abs_err)

