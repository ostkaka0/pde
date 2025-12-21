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

# Helper functions
def vecnorm(x):
  return np.sqrt(np.vecdot(x, x))

# Define the boundary curve for our domain
class OurCurve(bie.PolarCurve):
  def R(self, t):
    return (3 + np.cos(4*t + np.pi))
  def RPrim(self, t):
    return (-4 * np.sin(4*t + np.pi))
  def RBis(self, t):
    return (-16 * np.cos(4*t + np.pi))

if args.helm:
  def grad_phi(x):
    x_norm = vecnorm(x)[..., np.newaxis]
    return +1j/4 * args.k*x/x_norm * special.hankel1(1, args.k*x_norm)
else:
  def grad_phi(x):
    x_dot_x = np.vecdot(x, x)[..., np.newaxis]
    return 1/(2*np.pi) * x / x_dot_x

# Correct "secret" u at coord x
if args.helm:
  def secret_u(x):
    hval = special.hankel1(0, args.k * vecnorm(x - args.p))
    print(hval.dtype)
    return hval
else:
  def secret_u(x):
    return np.exp((x[...,0] + 0.3*x[...,1])/3) * np.sin((0.3*x[...,0] - x[...,1])/3)
  
def secret_v(x): # u at coord x
  return np.exp((x[...,0] + 0.3*x[...,1])/3) * np.cos((0.3*x[...,0] - x[...,1])/3)

## Plotting functions
# Plot A, B, and the log-abs error of A assuming B is the correct solution.
def plot_mat_comparison_and_show(A, B, extent=None, vmin=None, vmax=None):
  log_abs_err = np.log10(abs(A - B))
  fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)

  im0 = ax[0].imshow(A.T.real, origin = 'lower', vmin=vmin, vmax=vmax, extent=extent)
  im1 = ax[1].imshow(B.T.real, origin = 'lower', vmin=vmin, vmax=vmax, extent=extent)
  im2 = ax[2].imshow(log_abs_err.T, origin = 'lower', cmap='turbo', extent=extent)
  for a in ax:
    a.set_aspect('equal', adjustable='box')
    a.set_axis_off()
  plt.colorbar(im0)
  plt.colorbar(im1)
  plt.colorbar(im2)
  ax[0].set_title("approximation")
  ax[1].set_title("Correct solution")
  ax[2].set_title("Log-abs error")
  plt.tight_layout()
  plt.show()


# Excersises
curve = OurCurve()
t = np.linspace(-np.pi, np.pi, args.N, dtype=dtype, endpoint=False)
dt = 2*np.pi/args.N
dsdt = vecnorm(curve.rPrim(t))
x_bounds = (-4, 4, -4, 4)
t_bounds = (t[0], t[-1], t[0], t[-1])
x1 = np.linspace(x_bounds[0], x_bounds[1], args.M, dtype=dtype)
x2 = np.linspace(x_bounds[2], x_bounds[3], args.M, dtype=dtype)
X = np.stack(np.meshgrid(x1, x2), axis=-1)

# Used for exercise 2 and 3
t_odd = t - (t[1]-t[0])/2 # Assumes equal spacing between t-values
dsdt_odd = vecnorm(curve.rPrim(t_odd))
t_bounds_odd = (t_odd[0], t_odd[-1], t_odd[0], t_odd[-1])

## Problem 1:
if args.q == 1:
  print("Calculating kernel...")
  kernelMat = bie.calcKernelMat(t, grad_phi, curve);
  # plot_kernel_and_show(kernelMat, t_bounds, "Kernel matrix")
  title = f"Kernel matrix, N={args.N}"
  # Plot the kernel
  # Note that the kernel will be complex with helmholtz
  fig, ax = plt.subplots(1, 2 if np.iscomplexobj(kernelMat) else 1, squeeze = False)
  ax = ax[0]
  ax[0].set_title(title + ("(real)" if np.iscomplexobj(kernelMat) else ""))
  ax[0].set_xlabel("s")
  ax[0].set_ylabel("t")
  im1 = ax[0].imshow(kernelMat.real.T, origin = 'lower', cmap='turbo', extent=t_bounds)
  ax[0].axis('equal')
  fig.colorbar(im1)
  if np.iscomplexobj(kernelMat):
    ax[1].set_title(title + "(imag)")
    ax[1].set_xlabel("s")
    ax[1].set_ylabel("t")
    im2 = ax[1].imshow(kernelMat.T.imag, origin = 'lower', cmap='turbo', extent=t_bounds)
    ax[1].axis('equal')
    fig.colorbar(im2)
  plt.tight_layout()
  plt.show()

  print("Solving u...")
  g = secret_u(curve.r(t))
  u = bie.solve_u(kernelMat, X, t, dsdt, dt, g, grad_phi, curve)
  u_correct = secret_u(X) * curve.mask(X)
  plot_mat_comparison_and_show(u, u_correct, x_bounds, -1, 1)

## Problem 2:
kernelMat_odd = None
v = None
if args.q == 2 or args.q == 3:
  print("Calculating kernel...")
  kernelMat_odd = bie.calcKernelMat(t_odd, grad_phi, curve);
  g_odd = secret_u(curve.r(t_odd))
  v = bie.solve_boundary_v(t, t_odd, kernelMat_odd, dsdt_odd, dt, g_odd, curve)
  v_correct = secret_v(curve.r(t))

  if args.cheat:
    v = v_correct # We use the correct v from now on if we are cheating

  if args.q == 2:
    plt.plot(t, v, label="Approximation")
    plt.plot(t, v_correct, ":", label="Correct")
    plt.title("v at boundary")
    plt.show()
    # Plot log-abs-error
    plt.title("v at boundary - Log-abs error")
    log_abs_err_v = np.log10(abs(v - v_correct + np.mean(v_correct) - np.mean(v)))
    plt.plot(t_odd, log_abs_err_v)
    plt.show()

## Problem 3
print("Calculating odd kernel...")
kernelMat = bie.calcKernelMat(t_odd, grad_phi, curve);
if args.q == 3:
  g = secret_u(curve.r(t))
  u = bie.solve_u_better(args.M, t, g, v, x_bounds, curve)
  u_correct = secret_u(X) * curve.mask(X)
  plot_mat_comparison_and_show(u, u_correct, x_bounds, -3, 3)

# Problem 4
if args.q == 4:
  # Measure the far-field error for different N. We will just measure the error at the point(0, 0)
  x = np.array([0, 0])
  u_correct = secret_u(x)

  # Calculate u at (0, 0) for different Ns and calculate the error
  Ns        = np.arange(50, args.N + 10, 10)
  u         = np.zeros(Ns.shape, dtype=complex_dtype)
  err       = np.zeros(Ns.shape)
  for i, N in enumerate(tqdm(Ns)): 
    t = np.linspace(-np.pi, np.pi, N, dtype=dtype, endpoint=False)
    dsdt = vecnorm(curve.rPrim(t))
    dt = 2*np.pi/N
    kernelMat = bie.calcKernelMat(t, grad_phi, curve)
    g = secret_u(curve.r(t))
    u[i] = bie.solve_u(kernelMat, x, t, dsdt, dt, g, grad_phi, curve, show_progress=False)
    err[i] = abs(u[i] - u_correct)

  # Fit a curve to the error
  # log(y) = log(a + b log N) <=>
  # y = exp(a) * exp(b log N) <=>
  # y = exp(a) * N^b
  b, a = np.polyfit(np.log(Ns), np.log(err), 1)
  
  # Plot the error
  plt.show()
  plt.plot(Ns, err, label="Absolute error")
  plt.plot(Ns, np.exp(a) * Ns**b, label=f"{np.exp(a)} * N^{b}")
  # plt.xscale("log")
  plt.yscale("log")
  plt.xlabel("N")
  plt.legend()
  plt.show()
