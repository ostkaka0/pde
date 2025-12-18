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

def dot(x, y): return x[0]*y[0] + x[1]*y[1]
# def dot(x, y): return np.tensordot(x, y, axes=([0], [0]))
def abs_vec2(x):
  return sqrt(abs2_vec2(x))

## Helper functions
def abs2_vec2(x):
  return x[0]**2 + x[1]**2

def is_complex(x):
  return torch.is_complex(x) if args.pytorch else np.iscomplexobj(x)

def c2v(z):
  return np.stack((z.real, z.imag))

## Functions not specific to our problem
# Normal
def nu_complex(t):
  return -1j * rPrim_complex(t) / abs(rPrim_complex(t))
def nu(t):
  return c2v(nu_complex(t))





# TODO: def phi(x):

if args.helm: # phi_k
  def phi(x):
    return -1j/4 * special.hankel1(0, k * abs_vec2(x))
else: # regular phi
  def phi(x):
    return 1/(2*pi) * log(abs_vec2(x))

if args.helm:
  def grad_phi(x):
    return +1j/4 * k*(x)/abs_vec2(x) * special.hankel1(1, k*abs_vec2(x))
else:
  def grad_phi(x):
    return 1/(2*pi) * x/abs2_vec2(x)

# # The outgoing fundamental solution to the Helmholtz operator in 2 dimensions
# def phi_k(x):
# def grad_phi_k(x):


def kernel_non_diagonal(s, t):
  x, y = r(s), r(t)
  return dot(grad_phi(y-x), nu(t))
# def acoustic_kernel_non_diagonal(s, t):
#   x = r(s)
#   y = r(t)
#   return dot(grad_phi_k(y-x), nu(t))
def kernel_diagonal(t):
  return (
    1 / (4*pi)
    * (-RBis(t) * R(t)  + 2*RPrim(t)**2  + R(t)**2)
    / pow(RPrim(t)**2 + R(t)**2, 3/2)
  )
def calcKernelMat(t):
  T, S = np.meshgrid(t, t)
  mat = kernel_non_diagonal(S, T)
  diag = kernel_diagonal(t)

  # Insert the diagonals
  idcs = np.arange(len(t))
  mat[idcs, idcs] = diag

  return mat

# def calcKernelMat2(t):
#   T, S = np.meshgrid(t, t)
#   print(k)
#   mat = acoustic_kernel_non_diagonal(S, T)
#   diag = kernel_diagonal(t)

#   # Insert the diagonals
#   idcs = np.arange(len(t))
#   mat[idcs, idcs] = diag

#   return mat

def mask(x):
  t = np.atan2(x[0], x[1])
  return np.where(abs2_vec2(x) <= R(t)**2, 1, 0)
def mask_complex(z):
  return mask(np.array([z.real, z.imag]))

## BIE-algorithms
def solve_u(kernelMat, X, t, dsdt, dt):
  N = len(t)
  h = linalg.solve(eye(N)/2 + kernelMat @ diag(dsdt * dt), g(t))
  
  u = np.zeros(np.shape(X[0]), dtype=h.dtype)
  for i, t_i in enumerate(tqdm(t)):
    y_i = r(t_i)[:, None, None]
    nu_i = nu(t_i)[:, None, None]
   
    kernel_val = dot(grad_phi(y_i - X), nu(t_i))
    u += kernel_val * h[i] * dsdt[i] * dt

  return mask(X) * u

def correct_u(X):
  return mask(X) * secret_u(X)
  
def solve_boundary_v(t, t_odd, kernelMat_odd):
  N = len(t)
  dsdt_odd = abs_vec2(rPrim(t_odd))
  # kernelMat_odd = calcKernelMat(t_odd) 
  h_odd = linalg.solve(eye(N)/2 + 2*pi/N * kernelMat_odd @ diag(dsdt_odd), g(t_odd))

  dt = 2*pi/N
  x = r_complex(t)
  v = np.zeros((N), dtype=h_odd.dtype)
  
  for i, t_odd_i in enumerate(t_odd):
    y = r_complex(t_odd_i)
    numerator = nu_complex(t_odd_i)
    denominator = y - x
    phi = 1/(2*pi) * imag(numerator / denominator)
    v += phi * h_odd[i] * dsdt_odd[i] * dt
  return v

def solve_u_better(M, t, v, x_bounds):
  N = len(t)
  f = g(t) + 1j*v

  y = r_complex(t)
  dydt = rPrim_complex(t)

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

  return mask_complex(Z) * u

def correct_boundary_v(t):
  x = r(t)
  return secret_v(x)

## Plotting functions
def plot_mat_and_show(mat, extent=None, vmin=None, vmax=None):
  plt.imshow(real(mat.T), origin = 'lower', cmap='CMRmap_r', vmin=vmin, vmax=vmax, extent=extent)
  plt.axis('equal')
  plt.colorbar()
  plt.show()
def plot_kernel_and_show(mat, extent=None):
  plt.xlabel("s")
  plt.ylabel("t")
  plt.imshow(real(mat.T), origin = 'lower', cmap='CMRmap_r', extent=extent)
  plt.axis('equal')
  plt.colorbar()
  plt.show()
  if is_complex(mat):
    plt.title("imag")
    plt.xlabel("s")
    plt.ylabel("t")
    plt.imshow(imag(mat.T), origin = 'lower', cmap='CMRmap_r', extent=extent)
    plt.axis('equal')
    plt.colorbar()
    plt.show()
    

## Problem specific functions
def secret_u(r): # u at coord r
  if args.helm:
    # return phi_k(r - p)
    return special.hankel1(0, k * abs_vec2(r - p))
  else:
    return exp((r[0] + 0.3*r[1])/3) * sin((0.3*r[0] - r[1])/3)
  
def secret_v(r): # u at coord r
  return exp((r[0] + 0.3*r[1])/3) * cos((0.3*r[0] - r[1])/3)
# Boundary-values
def g(t):
  return secret_u(r(t))
# R(j) = distance from origin at time t
def R(t):
  return (3 + cos(4*t + pi))
def RPrim(t):
  return (-4 * sin(4*t + pi))
def RBis(t):
  return (-16 * cos(4*t + pi))
# r(t) = R exp(i t) = coord on boundary at time t
def r_complex(t):
  return R(t) * exp(1j * t)
def rPrim_complex(t):
  return (RPrim(t) + 1j*R(t)) * exp(1j * t)
def rBis_complex(t):
  return (RBis(t) + 2j*RPrim(t) - R(t)) * exp(1j * t)
def r(t): return c2v(r_complex(t))
def rPrim(t): return c2v(rPrim_complex(t))
def rBis(t): return c2v(rBis_complex(t))
# def r_vec(t):
#   return R(t) * np.stack([cos(t), sin(t)], axis=-1)
# def rPrim_vec(t):
#   return (diag(RPrim(t)) + 1j*R(t)) @ np.stack([cos(t), sin(t)], axis=-1)
# def rBis_vec(t):
#   return (RBis(t) + 2j*RPrim(t) - R(t)) * np.stack([cos(t), sin(t)], axis=-1)

### Excersises
t = np.linspace(-pi, pi, N, dtype=dtype, endpoint=False)
t_odd = t + (t[1]-t[0])/2 # Assumes equal spacing between t-values
dt = 2*pi/N
dsdt = abs_vec2(rPrim(t))
dsdt_odd = abs_vec2(rPrim(t_odd))
x_bounds = (-4, 4, -4, 4)
t_bounds = (t[0], t[-1], t[0], t[-1])
t_bounds_odd = (t_odd[0], t_odd[-1], t_odd[0], t_odd[-1])
x1 = np.linspace(x_bounds[0], x_bounds[1], M, dtype=dtype)
x2 = np.linspace(x_bounds[2], x_bounds[3], M, dtype=dtype)
X = np.array(np.meshgrid(x1, x2))

# plt.plot(t, kernel_diag())

## Problem 1:
print("Calculating kernel...")
kernelMat = calcKernelMat(t);
kernelMat_odd = calcKernelMat(t_odd);
if args.q == 1 or args.q == 0:
  plt.title("Kernel matrix")
  plot_kernel_and_show(kernelMat, t_bounds)
  print("Solving u...")
  u = solve_u(kernelMat, X, t, dsdt, dt)
  u_correct = correct_u(X)
  plt.title("")
  plt.xlabel("real x")
  plt.ylabel("imag x")
  plot_mat_and_show(u, x_bounds, -1, 1)
  plot_mat_and_show(u_correct, x_bounds, -1, 1)
  log_abs_err = log10(abs(u - u_correct))
  plot_mat_and_show(log_abs_err, x_bounds)

## Problem 2:
v = solve_boundary_v(t, t_odd, kernelMat_odd)
v_correct = correct_boundary_v(t)

if args.cheat:
  v = v_correct;

if args.q == 2 or args.q == 0:
  plt.plot(t, v)
  plt.plot(t, v_correct, ":")
  plt.show()
  # Plot log-abs-error
  log_abs_err_v = log10(abs(v - v_correct + np.mean(v_correct) - np.mean(v)))
  plt.plot(t_odd, log_abs_err_v)
  plt.show()

## Problem 3
print("Calculating odd kernel...")
kernelMat = calcKernelMat(t_odd);
if args.q == 3 or args.q == 0:
  plot_mat_and_show(kernelMat_odd, t_bounds_odd)
  u = solve_u_better(M, t, v, x_bounds)
  u_correct = correct_u(X)
  plot_mat_and_show(u, x_bounds, -3, 3)
  plot_mat_and_show(u_correct, x_bounds, -3, 3)
  log_abs_err = log10(abs(u - u_correct))
  plot_mat_and_show(log_abs_err, x_bounds)

## Problem 4:
if args.q == 4 or args.q == 0:
  d = np.linspace(0, 1.0)
  D, T = np.meshgrid(d, t)  
  X2 = r(T)- D * nu(T)
  # plot_mat_and_show(X2[1], None, -1, 1)

  plt.plot(X2[0, :], X2[1, :])
  plt.plot(X2[0, :, 10], X2[1, :, 10])
  plt.plot(X2[0, :,-1], X2[1, :, -1])
  plt.show()

  u2 = solve_u(kernelMat, X2, t, dsdt, dt)
  u2_correct = correct_u(X2)
  plot_mat_and_show(u2, None, -1, 1)
  plot_mat_and_show(u2_correct, None, -1, 1)
  log_abs_err = log10(abs(u2 - u2_correct))
  plot_mat_and_show(log_abs_err)

