import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_edges_3d_torch(n_x: int, n_y: int, n_z: int) -> torch.Tensor:
    vertices = torch.arange(n_x * n_y * n_z, device=device).reshape(n_x, n_y, n_z)
    edges_deep = torch.stack([vertices[:, :, :-1].reshape(-1), vertices[:, :, 1:].reshape(-1)])
    edges_right = torch.stack([vertices[:, :-1].reshape(-1), vertices[:, 1:].reshape(-1)])
    edges_down = torch.stack([vertices[:-1].reshape(-1), vertices[1:].reshape(-1)])
    edges = torch.cat([edges_deep, edges_right, edges_down], dim=1)
    return edges

def compute_gradients_3d_torch(data: torch.Tensor) -> torch.Tensor:
    gr_deep = torch.abs(data[:, :, :-1] - data[:, :, 1:]).reshape(-1)
    gr_right = torch.abs(data[:, :-1] - data[:, 1:]).reshape(-1)
    gr_down = torch.abs(data[:-1] - data[1:]).reshape(-1)
    return torch.cat([gr_deep, gr_right, gr_down])

def compute_weights_3d_torch(edges: torch.Tensor, data: torch.Tensor, beta: float = 130, eps: float = 1.e-6) -> torch.Tensor:
    l_x, l_y, l_z = data.shape
    gradients = compute_gradients_3d_torch(data)**2
    beta /= 10 * data.std()
    gradients *= beta
    weights = torch.exp(-gradients)
    weights += eps
    return gradients, beta, weights

def make_laplacian_sparse_torch(edges: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    pixel_nb = int(edges.max()) + 1
    pixel_nb = torch.tensor(pixel_nb, device=device)
    diag = torch.arange(pixel_nb, device=device)
    i_indices = torch.cat([edges[0], edges[1]])
    j_indices = torch.cat([edges[1], edges[0]])
    data = torch.cat([-weights, -weights])
    lap = torch.sparse_coo_tensor(
        torch.stack([i_indices, j_indices]),
        data,
        (pixel_nb, pixel_nb),
        device=device
    ).coalesce()
    connect = torch.sparse.sum(lap, dim=1).to_dense().neg()
    lap = torch.sparse_coo_tensor(
        torch.stack([torch.cat([i_indices, diag]), torch.cat([j_indices, diag])]),
        torch.cat([data, connect]),
        (pixel_nb, pixel_nb),
        device=device
    ).coalesce()

    return lap

def trim_edges_weights_torch(edges: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask0 = torch.cat([mask[:, :, :-1].reshape(-1), mask[:, :-1].reshape(-1), mask[:-1].reshape(-1)])
    mask1 = torch.cat([mask[:, :, 1:].reshape(-1), mask[:, 1:].reshape(-1), mask[1:].reshape(-1)])
    ind_mask = torch.logical_and(mask0, mask1)
    edges, weights = edges[:, ind_mask], weights[ind_mask]
    maxval = edges.max()
    order = torch.searchsorted(torch.unique(edges.view(-1)), torch.arange(maxval + 1, device=device))
    edges = order[edges]
    return edges, weights

def _build_laplacian_torch(data, mask=None, beta=50, eps=1.e-6):
    data = data.to(device)
    edges = make_edges_3d_torch(*data.shape)
    gradients, beta, weights = compute_weights_3d_torch(edges, data, beta, eps)
    if mask is not None:
        mask = mask.to(device)
        edges, weights = trim_edges_weights_torch(edges, weights, mask)
    lap = make_laplacian_sparse_torch(edges, weights)
    return lap

def random_walker_prior_torch(data, prior, mode='bf', gamma=1.e-2):
    device = data.device
    data = torch.atleast_3d(data).to(device)
    lap_sparse = _build_laplacian_torch(data, beta=50).to(device)
    shx, shy = lap_sparse.shape    
    if not lap_sparse.is_sparse:
        lap_sparse = lap_sparse.to_sparse()
    
    lap_sparse = lap_sparse + torch.sparse_coo_tensor(
                        torch.tensor([range(shx), range(shy)], device=device),
                        gamma*prior.sum(dim=0).to(device),
                        (shx, shy),
                        device=device)
    if mode == 'bf':
        lap_dense = lap_sparse.to_dense().double()        
        X = torch.stack([
            torch.linalg.solve(lap_dense, (gamma*label_prior.to(device)).double())
            for label_prior in prior
        ])
    return torch.squeeze((torch.argmax(X, dim=0)).reshape(data.shape))