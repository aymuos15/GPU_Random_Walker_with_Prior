import time

import numpy as np
import torch

from helper_numpy import _make_edges_3d
from helper_numpy import _compute_gradients_3d
from helper_numpy import _compute_weights_3d
from helper_numpy import _make_laplacian_sparse 
from helper_numpy import _trim_edges_weights
from helper_numpy import _build_laplacian
from helper_numpy import random_walker_prior

from helper_torch import make_edges_3d_torch
from helper_torch import compute_gradients_3d_torch
from helper_torch import compute_weights_3d_torch
from helper_torch import make_laplacian_sparse_torch
from helper_torch import trim_edges_weights_torch
from helper_torch import _build_laplacian_torch
from helper_torch import random_walker_prior_torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_x, n_y, n_z = 125, 125, 125
n_x_torch, n_y_torch, n_z_torch = torch.tensor(n_x, device=device), torch.tensor(n_y, device=device), torch.tensor(n_z, device=device)
cpu_time = []
gpu_time = []

data = np.random.rand(n_x, n_y, n_z)
data_torch = torch.tensor(data, device=device)

mask = np.ones_like(data, dtype=bool)
mask_torch = torch.ones_like(data_torch, dtype=torch.bool, device=device)

start_time = time.time()
edges = _make_edges_3d(n_x, n_y, n_z)
cpu_time.append(time.time() - start_time)

start_time = time.time()
edges_torch = make_edges_3d_torch(n_x_torch, n_y_torch, n_z_torch)
gpu_time.append(time.time() - start_time)

start_time = time.time()
_ = _compute_gradients_3d(data)
cpu_time.append(time.time() - start_time)

start_time = time.time()
_ = compute_gradients_3d_torch(data_torch)
gpu_time.append(time.time() - start_time)

start_time = time.time()
gradients, beta, weights = _compute_weights_3d(edges, data)
cpu_time.append(time.time() - start_time)

start_time = time.time()
gradients_torch, beta_torch, weights_torch = compute_weights_3d_torch(edges_torch, data_torch)
gpu_time.append(time.time() - start_time)

start_time = time.time()
lap = _make_laplacian_sparse(edges, weights)
cpu_time.append(time.time() - start_time)

start_time = time.time()
lap_torch = make_laplacian_sparse_torch(edges_torch, weights_torch)
gpu_time.append(time.time() - start_time)

start_time = time.time()
edges, weights = _trim_edges_weights(edges, weights, mask)
cpu_time.append(time.time() - start_time)

start_time = time.time()
edges_torch, weights_torch = trim_edges_weights_torch(edges_torch, weights_torch, mask_torch)
gpu_time.append(time.time() - start_time)

lap_numpy = _build_laplacian(data, mask)
lap_torch = _build_laplacian_torch(data_torch, mask_torch)

#### NIFTI MIMIC WALKER SEGMENTATION ####
a = np.zeros((15, 15, 15))
a[10:-10, 10:-10] = 1
a += 0.7 * np.random.random((a.shape))
a_flat = a.ravel()

prior = np.random.random((a.shape))  
prior_flat = prior.ravel()

n_phases = 2

prior_reshaped = np.zeros((n_phases, a.size))
prior_reshaped[0, :] = prior_flat  
prior_reshaped[1, :] = 1 - prior_flat  

a = torch.tensor(a)
prior = torch.tensor(prior_reshaped)

labs_numpy = random_walker_prior(a, prior_reshaped)
labs_torch = random_walker_prior_torch(a, prior)

print("CPU Time: ", sum(cpu_time))
print("GPU Time: ", sum(gpu_time))

total_gpu_speedup = -(sum(gpu_time) - sum(cpu_time)) / sum(cpu_time) * 100

print("Are the Laplacian results same?:", np.allclose(lap_numpy.data, lap_torch.values().cpu().numpy(), atol=1e-2))
print("GPU Speed up for lapcian @size: ", data.shape, total_gpu_speedup, "%")
print("Are the Walker results same?: @size",  a.shape, np.allclose(labs_numpy, labs_torch.cpu().numpy().reshape(a.shape)))