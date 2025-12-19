#!/usr/bin/python
# File created 2025-12-19 08:33:16 CET

import bie
import numpy as np
import argparse
from scipy import special
import matplotlib.pyplot as plt
from tqdm import tqdm

## Parse arguments
print("A")
parser = argparse.ArgumentParser()
print("B")
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
  default=1
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

## Set dtype based on argument
dtype = {
  32: np.float32,
  64: np.float64,
  128: np.float128
}[args.float]
complex_dtype = {
  32: np.complex64,
  64: np.complex128,
  128: np.complex256
}[args.float]
print("dtypes:", dtype, ",", complex_dtype)

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
  return np.iscomplexobj(x)
def vecnorm(x):
  return sqrt(vecdot(x, x))

## Define the boundary curve for our domain
class OurCurve(bie.PolarCurve):
  def R(self, t):
    return (3 + cos(4*t + np.pi))
  def RPrim(self, t):
    return (-4 * sin(4*t + np.pi))
  def RBis(self, t):
    return (-16 * cos(4*t + np.pi))

our_curve = OurCurve()

## Problem specific functions
# def phi_k(x):
#   return -1j/4 * special.hankel1(0, args.k * vecnorm(x))
# def phi(x):
#   return 1/(2*np.pi) * log(vecnorm(x))
if args.helm:
  def grad_phi(x):
    x_norm = vecnorm(x)[..., np.newaxis]
    return +1j/4 * args.k*x/x_norm * special.hankel1(1, args.k*x_norm)
else:
  def grad_phi(x):
    x_dot_x = vecdot(x, x)[..., np.newaxis]
    return 1/(2*np.pi) * x / x_dot_x

# Correct "secret" u at coord x
if args.helm:
  def secret_u(x):
    hval = special.hankel1(0, args.k * vecnorm(x - args.p))
    print(hval.dtype)
    return hval
else:
  def secret_u(x):
    return exp((x[...,0] + 0.3*x[...,1])/3) * sin((0.3*x[...,0] - x[...,1])/3)
  
def secret_v(x): # u at coord x
  return exp((x[...,0] + 0.3*x[...,1])/3) * cos((0.3*x[...,0] - x[...,1])/3)

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




### Excersises
curve = OurCurve()
t = np.linspace(-np.pi + 2*np.pi/args.N, np.pi, args.N, dtype=dtype)
# t = np.linspace(-np.pi, np.pi, args.N, dtype=dtype, endpoint=False)
t_odd = t - (t[1]-t[0])/2 # Assumes equal spacing between t-values
dt = 2*np.pi/args.N
dsdt = vecnorm(curve.rPrim(t))
dsdt_odd = vecnorm(curve.rPrim(t_odd))
x_bounds = (-4, 4, -4, 4)
t_bounds = (t[0], t[-1], t[0], t[-1])
t_bounds_odd = (t_odd[0], t_odd[-1], t_odd[0], t_odd[-1])
x1 = np.linspace(x_bounds[0], x_bounds[1], args.M, dtype=dtype)
x2 = np.linspace(x_bounds[2], x_bounds[3], args.M, dtype=dtype)
X = np.stack(np.meshgrid(x1, x2), axis=-1)

# plt.plot(t, kernel_diag())

## Problem 1:
print("Calculating kernel...")
kernelMat = bie.calcKernelMat(t, grad_phi, curve);
kernelMat_odd = bie.calcKernelMat(t_odd, grad_phi, curve);
if args.q == 1:
  plot_kernel_and_show(kernelMat, t_bounds, "Kernel matrix")
  print("Solving u...")
  g = secret_u(curve.r(t))
  u = bie.solve_u(kernelMat, X, t, dsdt, dt, g, grad_phi, curve)
  u_correct = secret_u(X) * curve.mask(X)
  plot_mat_comparison_and_show(u, u_correct, x_bounds, -1, 1)

## Problem 2:
g_odd = secret_u(curve.r(t_odd))
v = bie.solve_boundary_v(t, t_odd, kernelMat_odd, dsdt_odd, dt, g_odd, curve)
v_correct = secret_v(curve.r(t))
print("v", v.shape)
print("v_correct", v_correct.shape)
print("t", t.shape)
print("r t", curve.r(t).shape)

if args.cheat:
  v = v_correct

if args.q == 2:
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
kernelMat = bie.calcKernelMat(t_odd, grad_phi, curve);
if args.q == 3:
  plot_kernel_and_show(kernelMat_odd, t_bounds_odd, "Odd kernel matrix")
  g = secret_u(curve.r(t))
  u = bie.solve_u_better(args.M, t, g, v, x_bounds, curve)
  u_correct = secret_u(X) * curve.mask(X)
  plot_mat_comparison_and_show(u, u_correct, x_bounds, -3, 3)
  # plot_mat_and_show(u, x_bounds, -3, 3)
  # plot_mat_and_show(u_correct, x_bounds, -3, 3)
  # log_abs_err = log10(abs(u - u_correct))
  # plot_mat_and_show(log_abs_err, x_bounds)

# Problem 4
if args.q == 4:
  # Measure the far-field error for different N. We will just measure the error at the point(0, 0)
  x = np.array([0, 0])

  Ns        = np.arange(20, 500, 20)
  u         = np.zeros(Ns.shape, dtype=complex_dtype)
  u_correct = np.zeros(Ns.shape, dtype=complex_dtype)
  err       = np.zeros(Ns.shape)
  for i, N2 in enumerate(tqdm(Ns)): 
    t = np.linspace(-np.pi, np.pi, N2, dtype=dtype, endpoint=False)
    dsdt = vecnorm(curve.rPrim(t))
    kernelMat = bie.calcKernelMat(t, grad_phi, curve)
    g = secret_u(curve.r(t))
    u[i] = bie.solve_u(kernelMat, x, t, dsdt, dt, g, grad_phi, curve, show_progress=False)
    u_correct[i] = secret_u(x) * curve.mask(x)
    err[i] = log10(abs(u[i] - u_correct[i]))

  plt.plot(Ns, real(u), label="real u")
  plt.plot(Ns, imag(u), label="imag u")
  plt.plot(Ns, real(u_correct), ":", label="real u_correct")
  plt.plot(Ns, imag(u_correct), ":", label="imag u_correct")
  plt.plot(Ns, err, label="err")
  plt.legend()
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

