import numpy as np
from scipy import sparse
try:
    import numexpr as ne
    numexpr_loaded = True
except ImportError:
    numexpr_loaded = False

def _make_edges_3d(n_x, n_y, n_z):
    vertices = np.arange(n_x * n_y * n_z).reshape((n_x, n_y, n_z))
    edges_deep = np.vstack((vertices[:, :, :-1].ravel(),
                            vertices[:, :, 1:].ravel()))
    edges_right = np.vstack((vertices[:, :-1].ravel(), vertices[:, 1:].ravel()))
    edges_down = np.vstack((vertices[:-1].ravel(), vertices[1:].ravel()))
    edges = np.hstack((edges_deep, edges_right, edges_down))
    return edges

def _compute_gradients_3d(data):
    l_x, l_y, l_z = data.shape
    gr_deep = np.abs(data[:, :, :-1] - data[:, :, 1:]).ravel()
    gr_right = np.abs(data[:, :-1] - data[:, 1:]).ravel()
    gr_down = np.abs(data[:-1] - data[1:]).ravel()
    return np.r_[gr_deep, gr_right, gr_down]

def _compute_weights_3d(edges, data, beta=130, eps=1.e-6):
    l_x, l_y, l_z = data.shape
    gradients = _compute_gradients_3d(data)**2
    beta /= 10 * data.std()
    gradients *= beta
    if numexpr_loaded:
        weights = ne.evaluate("exp(- gradients)")
    else:
        weights = np.exp(- gradients)
    weights += eps

    return gradients, beta, weights

def _make_laplacian_sparse(edges, weights):
    pixel_nb = edges.max() + 1
    diag = np.arange(pixel_nb)
    i_indices = np.hstack((edges[0], edges[1]))
    j_indices = np.hstack((edges[1], edges[0]))
    data = np.hstack((-weights, -weights))
    lap = sparse.coo_matrix((data, (i_indices, j_indices)),
                            shape=(pixel_nb, pixel_nb))
    connect = - np.ravel(lap.sum(axis=1))
    lap = sparse.coo_matrix((np.hstack((data, connect)),
                (np.hstack((i_indices,diag)), np.hstack((j_indices, diag)))),
                shape=(pixel_nb, pixel_nb))
    return lap.tocsr()

def _trim_edges_weights(edges, weights, mask):
    mask0 = np.hstack((mask[:, :, :-1].ravel(), mask[:, :-1].ravel(), mask[:-1].ravel()))
    mask1 = np.hstack((mask[:, :, 1:].ravel(), mask[:, 1:].ravel(), mask[1:].ravel()))
    ind_mask = np.logical_and(mask0, mask1)
    edges, weights = edges[:, ind_mask], weights[ind_mask]
    maxval = edges.max()
    order = np.searchsorted(np.unique(edges.ravel()), np.arange(maxval+1))
    edges = order[edges]
    return edges, weights

def _build_laplacian(data, mask=None, beta=50, eps=1.e-6):
    l_x, l_y, l_z = data.shape
    edges = _make_edges_3d(l_x, l_y, l_z)
    gradients, beta, weights = _compute_weights_3d(edges, data, beta=beta, eps=1.e-6)
    if mask is not None:
        edges, weights = _trim_edges_weights(edges, weights, mask)
    lap =  _make_laplacian_sparse(edges, weights)
    return lap

def random_walker_prior(data, prior, mode='bf', gamma=1.e-2):
    data = np.atleast_3d(data)
    lap_sparse = _build_laplacian(data, beta=50)
    dia = range(data.size)
    shx, shy = lap_sparse.shape
    lap_sparse = lap_sparse + sparse.coo_matrix(
                        (gamma*prior.sum(axis=0), (range(shx), range(shy))))
    del dia
    if mode == 'bf':
        lap_sparse = lap_sparse.tocsc()
        solver = sparse.linalg.factorized(lap_sparse.astype(np.double))
        X = np.array([solver(gamma*label_prior)
                      for label_prior in prior])
    return np.squeeze((np.argmax(X, axis=0)).reshape(data.shape))