#!/usr/bin/python
# File created 2025-12-09 15:19:32 CET

# h - density
# u - electrostatic potential
# u = g at boundary(known)
# u is a weighted average of fundamental solutions with poles on boundary.
# we choose h so that u = g at boundary

# 


 
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
pi = np.pi
abs = np.abs # Note: Overrides built-in abs
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
dot = np.dot

## Helper functions
def abs2(z):
  return real(z * conj(z))
def to_complex(z):
  if not args.pytorch:
    return z
  else:
    return z.to(complex_dtype)

## Functions not specific to our problem
# Normal
def nu(t):
  return -1j * rPrim(t) / abs(rPrim(t))

def kernel_non_diagonal(s, t):
  x = r(s)
  y = r(t)
  return 1/(2*pi) * real((y-x)*conj(nu(t))) / abs2(y-x)


p = 3


# TODO: def phi(x):

# The outgoing fundamental solution to the Helmholtz operator in 2 dimensions
def phi_k(x, k):
  -1j/4 * special.hankel0(1, k * abs(x))
def grad_phi_k(x, k):
  return 1j/4 * k*x/abs(x) * special.hankel1(1, k*abs(x))


def acoustic_kernel_non_diagonal(s, t, k):
  x = r(s)
  y = r(t)
  return real(grad_phi_k(y-x, k)*conj(nu(t)))
def kernel_diagonal(t):
  return (
    1 / (4*pi)
    * (-RBis(t) * R(t)  + 2*RPrim(t)**2  + R(t)**2)
    / pow(RPrim(t)**2 + R(t)**2, 3/2)
  )
def calcKernelMat(t):
  s = t[:, None]
  mat = kernel_non_diagonal(s, t)
  diag = kernel_diagonal(t)

  # Insert the diagonals
  idcs = np.arange(len(t))
  mat[idcs, idcs] = diag

  return mat
def calcKernelMat2(t, k):
  s = t[:, None]
  print(k)
  mat = acoustic_kernel_non_diagonal(s, t, k)
  diag = kernel_diagonal(t)

  # Insert the diagonals
  idcs = np.arange(len(t))
  mat[idcs, idcs] = diag

  return mat

def mask(z):
  t = np.atan2(imag(z), real(z))
  return np.where(abs2(z) <= R(t)**2, 1, 0)

## BIE-algorithms
def solve_u(M, t, bounds, kernelMat):
  N = len(t)
  # dsdt = sqrt(RPrim(t)**2 + R(t)**2)
  dsdt = abs(rPrim(t))
  h = linalg.solve(eye(N)/2 + 2*pi/N*kernelMat@diag(dsdt), g(t))

  x1 = np.linspace(bounds[0], bounds[1], M, dtype=dtype)[:, None]
  x2 = np.linspace(bounds[2], bounds[3], M, dtype=dtype)[None, :]
  x = x1 + 1j*x2
  
  u = np.zeros((M, M), dtype=dtype)
  
  for i, t_i in enumerate(tqdm(t)):
    y = r(t_i)
    numerator = real(nu(t_i)*conj(y - x))
    denominator = abs2(y - x)
    phi = 1/(2*pi) * numerator / denominator
    u += phi * h[i] * dsdt[i] * 2*pi/N

  return mask(x) * u

def correct_u(M, bounds):
  x1 = np.linspace(bounds[0], bounds[1], M, dtype=dtype)[:, None]
  x2 = np.linspace(bounds[2], bounds[3], M, dtype=dtype)[None, :]
  x = x1 + 1j*x2
  u = None
  if args.helm:
    u = secret_u2(x)
  else:
    u = secret_u(x)
  return mask(x) * u
  
def solve_boundary_v(t, t_odd, kernelMat_odd):
  N = len(t)
  dsdt_odd = abs(rPrim(t_odd))
  # kernelMat_odd = calcKernelMat(t_odd) 
  h_odd = linalg.solve(eye(N)/2 + 2*pi/N * kernelMat_odd @ diag(dsdt_odd), g(t_odd))

  x = r(t)
  v = np.zeros((N))
  for i, t_odd_i in enumerate(t_odd):
    y = r(t_odd_i)
    numerator = nu(t_odd_i)
    denominator = y - x
    phi = 1/(2*pi) * imag(numerator / denominator)
    v += phi * h_odd[i] * dsdt_odd[i] * 2*pi/N
  return v

def solve_u_better(M, t, v, bounds):
  N = len(t)
  f = g(t) + 1j*v

  y = r(t)
  dydt = rPrim(t)

  x1 = np.linspace(bounds[0], bounds[1], M, dtype=dtype)[:, None]
  x2 = np.linspace(bounds[2], bounds[3], M, dtype=dtype)[None, :]
  x = x1 + 1j*x2
  
  numerator = np.zeros((M, M), dtype=complex_dtype)
  denominator = np.zeros((M, M), dtype=complex_dtype)
  for i, t_i in enumerate(tqdm(t)):
    numerator   += (f[i] / (y[i]-x)) * dydt[i] * 2*pi/N
    denominator += (1    / (y[i]-x)) * dydt[i] * 2*pi/N
  u = real(numerator / denominator)

  return mask(x) * u

def correct_boundary_v(t):
  x = r(t)
  return secret_v(x)

## Plotting functions
def plot_mat_and_show(mat, extent=None, vmin=None, vmax=None):
  plt.imshow(mat.T, origin = 'lower', cmap='CMRmap_r', vmin=vmin, vmax=vmax, extent=extent)
  plt.axis('equal')
  plt.colorbar()
  plt.show()

## Problem specific functions
def secret_u(r): # u at coord r
  return exp((r.real + 0.3*r.imag)/3) * sin((0.3*r.real - r.imag)/3)
def secret_u2(r): # u at coord r
  return special.hankel1(0, k * abs(r - 3))
def secret_v(r): # u at coord r
  return exp((r.real + 0.3*r.imag)/3) * cos((0.3*r.real - r.imag)/3)
# Boundary-values
def g(t):
  if args.helm:
    return secret_u2(r(t))
  else:
    return secret_u(r(t))
# R(j) = distance from origin at time t
def R(t):
  return 3 + cos(4*t + pi)
def RPrim(t):
  return -4 * sin(4*t + pi)
def RBis(t):
  return -16 * cos(4*t + pi)
# r(t) = R exp(i t) = coord on boundary at time t
def r(t):
  return R(t) * exp(1j * to_complex(t))
def rPrim(t):
  return (RPrim(t) + 1j*R(t)) * exp(1j * to_complex(t))
def rBis(t):
  return (RBis(t) + 2j*RPrim(t) - R(t)) * exp(1j * to_complex(t))

### Excersises
t = np.linspace(-pi + 2*pi/N, pi, N, dtype=dtype)
t_odd = t + (t[1]-t[0])/2 # Assumes equal spacing between t-values
bounds = (-4, 4, -4, 4)
t_bounds = (t[0], t[-1], t[0], t[-1])
t_bounds_odd = (t_odd[0], t_odd[-1], t_odd[0], t_odd[-1])

# plt.plot(t, kernel_diag())

## Problem 1:
print("Calculating kernel...")
kernelMat = None
kernelMat_odd = None
k = 1
if args.helm:
  kernelMat = calcKernelMat2(t, k);
  kernelMat_odd = calcKernelMat2(t_odd, k);
else:
  kernelMat = calcKernelMat(t);
  kernelMat_odd = calcKernelMat(t_odd);
plt.title("Kernel matrix")
plt.xlabel("s")
plt.ylabel("t")
plot_mat_and_show(kernelMat, t_bounds)
if args.q == 1 or args.q == 0:
  print("Solving u...")
  u = solve_u(M, t, bounds, kernelMat)
  u_correct = correct_u(M, bounds)
  plt.title("")
  plt.xlabel("real x")
  plt.ylabel("imag x")
  plot_mat_and_show(u, bounds, -3, 3)
  plot_mat_and_show(u_correct, bounds, -3, 3)
  log_abs_err = log10(abs(u - u_correct))
  plot_mat_and_show(log_abs_err, bounds)

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
plot_mat_and_show(kernelMat_odd, t_bounds_odd)
if args.q == 3 or args.q == 0:
  u = solve_u_better(M, t, v, bounds)
  u_correct = correct_u(M, bounds)
  plot_mat_and_show(u, bounds, -3, 3)
  plot_mat_and_show(u_correct, bounds, -3, 3)
  log_abs_err = log10(abs(u - u_correct))
  plot_mat_and_show(log_abs_err, bounds)

## Problem 4:
if args.q == 4 or args.q == 0:
  print("Calculating kernel... <<<")
  kernelMat = calcKernelMat2(t, k);
  plot_mat_and_show(kernelMat, bounds)

