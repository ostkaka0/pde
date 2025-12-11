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
  default=300
)
parser.add_argument(
  "-M",
  type=int,
  default=800
)
parser.add_argument(
  "--float",
  type=int,
  default=64
)
args = parser.parse_args()

## Imports
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
def kernel(s, t):
  x = r(s)
  y = r(t)
  if x != y:
    return 1/(2*pi) * real((y-x)*conj(nu(t))) / abs(y-x)**2
  else:
    return (
      +1 / (4*pi)
      * (-RBis(t) * R(t)  + 2*RPrim(t)**2  + R(t)**2)
      / pow(RPrim(t)**2 + R(t)**2, 3/2)
    )

## Problem specific functions
def secret_solution(r): # u at coord r
  return exp((r.real + 0.3*r.imag)/3) * sin((0.3*r.real - r.imag)/3)
# Boundary-values
def g(t):
  return secret_solution(r(t))
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

## BIE
# Common linspaces and meshgrids:
t = torch.linspace(-pi + 2*pi/N, pi, N, dtype=dtype)
xVec = torch.linspace(-4, 4, M, dtype=dtype)
yVec = torch.linspace(-4, 4, M, dtype=dtype)
X, Y = torch.meshgrid(xVec, yVec)
Z = X + 1j*Y

# Plot boundary, rPrim and nu(normals)
plt.title("Boundary, r' and ν")
plt.plot(r(t).real, r(t).imag)
plt.quiver(r(t).real, r(t).imag, 1.0*rPrim(t).real, 1.0*rPrim(t).imag, width=0.002, color='r')
plt.quiver(r(t).real, r(t).imag, 1.0*nu(t).real, 1.0*nu(t).imag, width=0.002, color='g')
plt.show()

# Calculate kernel 
kernelMat = torch.zeros((N, N))
for i, x in enumerate(tqdm(t)):
  for j, y in enumerate(t):
    kernelMat[i, j] = kernel(t[i], t[j])
# Plot kernel
plt.figure()
plt.imshow(kernelMat.T, origin = 'lower', cmap='CMRmap_r')
plt.axis('equal')
plt.colorbar()
plt.show()

dsdt = sqrt(RPrim(t)**2 + R(t)**2)
h = torch.linalg.solve(eye(N)/2 + 2*pi/N*kernelMat@diag(dsdt), g(t))

# Calculate u by BIE, and also we do know the correct one.
uField = torch.zeros((M, M), dtype=dtype)
uCorrect = torch.zeros((M, M), dtype=dtype)
for i, x1 in enumerate(tqdm(xVec)):
  for j, x2 in enumerate(yVec):
    x = x1 + 1j*x2
    y = r(t)
    tt = torch.atan2(x2, x1)
    if x1**2+x2**2 <= R(tt)**2:
      numerator = real(nu(t)*conj(y - x))
      denominator = abs2(y - x)
      phi = 1/(2*pi) * numerator / denominator
      uField[i, j] = torch.dot(phi, h * dsdt)*2*pi/N
      uCorrect[i, j] = secret_solution(x)

# Plot BIE-solution
plt.figure()
plt.imshow(uField.T, origin = 'lower', cmap='CMRmap_r', vmin=-3, vmax=3)
plt.axis('equal')
plt.colorbar()
plt.show()
# Plot correct solution
plt.figure()
plt.imshow(uCorrect.T, origin = 'lower', cmap='CMRmap_r', vmin=-3, vmax=3)
plt.axis('equal')
plt.colorbar()
plt.show()
# Plot log-abs-error
log_abs_err = log10(abs(uField - uCorrect))
plt.figure()
plt.imshow(log_abs_err.T, origin = 'lower', cmap='CMRmap_r')
plt.axis('equal')
plt.colorbar()
plt.show()
