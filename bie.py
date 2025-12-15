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
  default=100
)
parser.add_argument(
  "-M",
  type=int,
  default=100
)
parser.add_argument(
  "--float",
  type=int,
  default=64
)
args = parser.parse_args()

## Imports
import time
if not args.pytorch:
  import numpy as torch
else:
  import torch
import matplotlib.pyplot as plt
from tqdm import tqdm # Progress bar
# from scipy.special import hankel1

## Constants
N = args.N
M = args.M

## Set correct dtype
dtype = {
  32: torch.float32,
  64: torch.float64,
}[args.float]
complex_dtype = {
  32: torch.complex64,
  64: torch.complex128,
}[args.float]
print("dtypes:", dtype, ",", complex_dtype)

if args.pytorch:
  print("Using pytorch")
  print("CPU threads used by pytorch:", torch.get_num_threads())

## Aliases
pi = torch.pi
abs = torch.abs # Note: Overrides built-in abs
pow = torch.pow
exp = torch.exp
cos = torch.cos
sin = torch.sin
sqrt = torch.sqrt
real = torch.real
imag = torch.imag
conj = torch.conj
log = torch.log
log10 = torch.log10
eye = torch.eye
diag = torch.diag
dot = torch.dot
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
  return 1/(2*pi) * real((y-x)*conj(nu(t))) / abs(y-x)**2

def kernel_diagonal(t):
  return (
    1 / (4*pi)
    * (-RBis(t) * R(t)  + 2*RPrim(t)**2  + R(t)**2)
    / pow(RPrim(t)**2 + R(t)**2, 3/2)
  )

def kernel_element(s, t):
  if s != t:
    return kernel_non_diagonal(s, t);
  else:
    return kernel_diagonal(t);

def calcKernelMat(t):
  s = t[:, None]
  mat = kernel_non_diagonal(s, t)
  diag = kernel_diagonal(t)

  # Insert the diagonals
  idcs = torch.arange(len(t))
  mat[idcs, idcs] = diag

  return mat

def calcKernelMatSlow(t):
  mat = torch.zeros((N, N))
  for i, x in enumerate(tqdm(t)):
    for j, y in enumerate(t):
      mat[i, j] = kernel_element(t[i], t[j])
  return mat

def mask(z):
  tt = torch.atan2(imag(z), real(z))
  return torch.where(abs2(z) <= R(tt)**2, 1, 0)

## Problem specific functions
def secret_u(r): # u at coord r
  return exp((r.real + 0.3*r.imag)/3) * sin((0.3*r.real - r.imag)/3)
def secret_v(r): # u at coord r
  return exp((r.real + 0.3*r.imag)/3) * cos((0.3*r.real - r.imag)/3)
# Boundary-values
def g(t):
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

## BIE-algorithms
def solve_u(M, t, bounds):
  N = len(t)
  # dsdt = sqrt(RPrim(t)**2 + R(t)**2)
  dsdt = abs(rPrim(t))
  h = torch.linalg.solve(eye(N)/2 + 2*pi/N*kernelMat@diag(dsdt), g(t))

  x1 = torch.linspace(*bounds[0], M, dtype=dtype)[:, None]
  x2 = torch.linspace(*bounds[1], M, dtype=dtype)[None, :]
  x = x1 + 1j*x2
  
  u = torch.zeros((M, M), dtype=dtype)
  
  for i, t_i in enumerate(tqdm(t)):
    y = r(t_i)
    numerator = real(nu(t_i)*conj(y - x))
    denominator = abs2(y - x)
    phi = 1/(2*pi) * numerator / denominator
    u += phi * h[i] * dsdt[i] * 2*pi/N

  return mask(x) * u

def correct_u(M, bounds):
  x1 = torch.linspace(*bounds[0], M, dtype=dtype)[:, None]
  x2 = torch.linspace(*bounds[1], M, dtype=dtype)[None, :]
  x = x1 + 1j*x2
  u = secret_u(x)
  return mask(x) * u
  
def solve_boundary_v(t, t_odd):
  dsdt_odd = abs(rPrim(t_odd))
  h_odd = torch.linalg.solve(eye(N)/2 + 2*pi/N*kernelMat@diag(dsdt_odd), g(t_odd))
  kernelMat_odd = calcKernelMat(t_odd) 

  x = r(t)[:, None]
  y = r(t_odd)
  numerator = nu(t_odd)
  denominator = y - x
  phi = 1/(2*pi) * imag(numerator / denominator)
  v = torch.sum(phi * h_odd * dsdt_odd, axis=-1)*2*pi/N
  return v

def solve_u_better(M, t, v, bounds):
  N = len(t)
  f = g(t) + 1j*v

  y = r(t)
  dydt = rPrim(t)

  x1 = torch.linspace(*bounds[0], M, dtype=dtype)[:, None]
  x2 = torch.linspace(*bounds[1], M, dtype=dtype)[None, :]
  z = x1 + 1j*x2
  
  
  numerator = torch.zeros((M, M), dtype=complex_dtype)
  denominator = torch.zeros((M, M), dtype=complex_dtype)
  for i, t_i in enumerate(tqdm(t)):
    numerator   += (f[i] / (y[i]-z)) * dydt[i] * 2*pi/N
    denominator += (1    / (y[i]-z)) * dydt[i] * 2*pi/N
  u = real(numerator / denominator)

  return mask(z) * u

def correct_boundary_v(t):
  x = r(t)
  return secret_v(x)

## Plotting functions
def plot_mat_and_show(mat, vmin=None, vmax=None):
  plt.imshow(mat.T, origin = 'lower', cmap='CMRmap_r', vmin=vmin, vmax=vmax)
  plt.axis('equal')
  plt.colorbar()
  plt.show()

## BIE
# Common linspaces and meshgrids:
t = torch.linspace(-pi + 2*pi/N, pi, N, dtype=dtype)
xVec = torch.linspace(-4, 4, M, dtype=dtype)
yVec = torch.linspace(-4, 4, M, dtype=dtype)

X, Y = torch.meshgrid(xVec, yVec)
Z = X + 1j*Y

print("Calculating kernel...")
kernelMat = calcKernelMat(t);
plot_mat_and_show(kernelMat)

## Problem 1:
bounds = ((-4, 4), (-4, 4))
print("Solving u...")
u = solve_u(M, t, bounds)
u_correct = correct_u(M, bounds)
plot_mat_and_show(u, -3, 3)
plot_mat_and_show(u_correct, -3, 3)
log_abs_err = log10(abs(u - u_correct))
plot_mat_and_show(log_abs_err)

## Problem 2:
t_odd = t + (t[1]-t[0])/2 # Assumes equal spacing between t-values
v = solve_boundary_v(t, t_odd)
v_correct = correct_boundary_v(t)
plt.plot(t, v)
plt.plot(t, v_correct, ":")
plt.show()
# Plot log-abs-error
v2 = v + torch.mean(v_correct - v);
log_abs_err_v = log10(abs(v - v_correct))
plt.plot(t_odd, log_abs_err_v)
plt.show()
v = v2
v = v_correct;

## Problem 3
u2 = solve_u_better(M, t, v, bounds)

# Plot BIE-solution
plt.figure()
plt.imshow(u2.T, origin = 'lower', cmap='CMRmap_r', vmin=-3, vmax=3)
plt.colorbar()
plt.show()
# Plot correct solution
plt.figure()
plt.imshow(u_correct.T, origin = 'lower', cmap='CMRmap_r', vmin=-3, vmax=3)
plt.colorbar()
plt.show()
# Plot log-abs-error
log_abs_err = log10(abs(u2 - u_correct))
plt.figure()
plt.imshow(log_abs_err.T, origin = 'lower', cmap='CMRmap_r')
plt.colorbar()
plt.show()
