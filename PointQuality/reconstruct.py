# edgerecon.py
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
from sklearn.manifold import MDS
from sklearn import manifold  # used in landmark MDS step
import networkx as nx

# Optional dependencies (required only for specific modes)
try:
    import umap
except Exception:
    umap = None
try:
    import torch, pymde
except Exception:
    torch = None
    pymde = None


# ------------------------------- I/O helpers -------------------------------- #

def _load_edge_list(edge_list):
    """
    Accepts: CSV path (str), DataFrame, or ndarray with columns [source, target(, weight)].
    Returns: (src, dst, w, node_labels) with nodes as original labels.
    """
    if isinstance(edge_list, str):
        df = pd.read_csv(edge_list)
    elif isinstance(edge_list, pd.DataFrame):
        df = edge_list.copy()
    elif isinstance(edge_list, np.ndarray):
        if edge_list.ndim != 2 or edge_list.shape[1] < 2:
            raise ValueError("NumPy edge list must have shape (M,2) or (M,3).")
        cols = ["source", "target"] + (["weight"] if edge_list.shape[1] >= 3 else [])
        df = pd.DataFrame(edge_list, columns=cols)
    else:
        raise TypeError("edge_list must be a CSV path, pandas DataFrame, or numpy array.")

    # normalize column names
    df.columns = [c.lower() for c in df.columns]
    if not {"source", "target"}.issubset(df.columns):
        raise ValueError("Edge list must have columns 'source' and 'target' (and optional 'weight').")

    src = df["source"].to_numpy()
    dst = df["target"].to_numpy()
    if "weight" in df.columns:
        w = df["weight"].to_numpy(dtype=float)
    else:
        w = np.ones(len(src), dtype=float)

    node_labels = np.unique(np.concatenate([src, dst]))
    return src, dst, w, node_labels


def _to_sparse_adjacency(src, dst, w, node_labels, undirected=True):
    """Build a CSR adjacency and preserve original node labels."""
    idx = {n: i for i, n in enumerate(node_labels)}
    rows = np.vectorize(idx.get)(src)
    cols = np.vectorize(idx.get)(dst)
    n = len(node_labels)

    A = sp.coo_matrix((w, (rows, cols)), shape=(n, n))
    if undirected:
        # symmetrize and zero diagonal
        A = A + A.T
        A.setdiag(0)
        A.eliminate_zeros()
    return A.tocsr()


# ---------------------------- Shortest paths helper -------------------------- #

def _all_pairs_shortest_paths(A: sp.spmatrix, *, unweighted=False) -> np.ndarray:
    """Compute all-pairs shortest paths from a sparse adjacency."""
    return shortest_path(A, directed=False, unweighted=unweighted)


# ------------------------------- Core API ----------------------------------- #

def reconstruct(
    edge_list,
    *,
    mode: str = "MDS",
    dim: int = 2,
    shortest_path_matrix: np.ndarray | None = None,
    n_landmarks: int = 256,
    n_neighbors: int = 15,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct node coordinates from an edge list.

    Parameters
    ----------
    edge_list : str | DataFrame | ndarray
        CSV path, DataFrame, or array with columns [source, target(, weight)].
    mode : str
        One of: "MDS", "classical_MDS", "landmark_isomap", "landmark_isomap_weighted",
                 "spring_relaxation", "umap", "PyMDE", "PyMDE_weighted".
    dim : int
        Output dimensionality (typically 2 or 3).
    shortest_path_matrix : ndarray | None
        Optionally precomputed N x N distances. If None, computed as needed.
        For "umap" with metric='precomputed', this must be distances.
    n_landmarks : int
        #landmarks for landmark-based modes.
    n_neighbors : int
        Neighborhood size for UMAP (if used).
    random_state : int
        Seed for deterministic behavior where supported.

    Returns
    -------
    positions : np.ndarray, shape (N, dim)
    node_ids : np.ndarray, shape (N,)
        node_ids[i] is the original node label corresponding to positions[i].
    """
    # Load + build adjacency
    src, dst, w, node_labels = _load_edge_list(edge_list)
    A = _to_sparse_adjacency(src, dst, w, node_labels, undirected=True)

    # Detect 'weighted' (any non-1.0 weight)
    is_weighted = A.nnz > 0 and not np.allclose(A.data, 1.0)

    # Dispatch by mode
    mode = mode.strip()

    if mode == "spring_relaxation":
        G = nx.from_scipy_sparse_array(A)
        pos_dict = nx.spring_layout(G, seed=random_state, dim=dim)
        X = np.array([pos_dict[i] for i in range(A.shape[0])])
        return X, node_labels

    if mode == "MDS":
        D = shortest_path_matrix if shortest_path_matrix is not None else _all_pairs_shortest_paths(A, unweighted=False)
        mds = MDS(n_components=dim, dissimilarity="precomputed", random_state=random_state)
        X = mds.fit_transform(D)
        return X, node_labels

    if mode == "classical_MDS":
        D = shortest_path_matrix if shortest_path_matrix is not None else _all_pairs_shortest_paths(A, unweighted=False)
        n = D.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H @ (D ** 2) @ H
        w_eig, V = np.linalg.eigh(B)
        idx = np.argsort(-w_eig)[:dim]
        L = np.diag(np.sqrt(np.maximum(w_eig[idx], 0)))
        X = V[:, idx] @ L
        return X, node_labels

    if mode == "umap":
        if umap is None:
            raise ImportError("UMAP mode requires 'umap-learn'. Try: pip install umap-learn")
        D = shortest_path_matrix if shortest_path_matrix is not None else _all_pairs_shortest_paths(A, unweighted=False)
        reducer = umap.UMAP(
            n_components=dim, n_neighbors=n_neighbors, min_dist=1.0, metric="precomputed", random_state=random_state
        )
        X = reducer.fit_transform(D)
        return X, node_labels

    if mode == "landmark_isomap":
        # Unweighted BFS distances to landmarks
        X = _landmark_isomap_unweighted(A, dim=dim, n_landmarks=n_landmarks, random_state=random_state)
        return X, node_labels

    if mode == "landmark_isomap_weighted":
        # Use weighted shortest paths if available/provided
        D = shortest_path_matrix if shortest_path_matrix is not None else _all_pairs_shortest_paths(A, unweighted=False)
        X = _landmark_isomap_from_distances(D, dim=dim, n_landmarks=n_landmarks, random_state=random_state)
        return X, node_labels

    if mode == "PyMDE" or (mode == "PyMDE_weighted"):
        if pymde is None or torch is None:
            raise ImportError('PyMDE modes require "torch" and "pymde". Try: pip install torch pymde')
        Acoo = A.tocoo()
        rows, cols, weights = Acoo.row, Acoo.col, Acoo.data.astype(float)
        if mode == "PyMDE_weighted" and is_weighted:
            distances = np.where(np.abs(weights) < 1e-12, 1e-12, weights) ** (-1)
            edges = torch.tensor(np.vstack([rows, cols]).T, dtype=torch.int64)
            w_t = torch.tensor(distances, dtype=torch.float32)
            g = pymde.Graph.from_edges(edges, weights=w_t)
        else:
            edges = torch.tensor(np.vstack([rows, cols]).T, dtype=torch.int64)
            g = pymde.Graph.from_edges(edges, weights=None)

        spg = pymde.preprocess.graph.shortest_paths(g, retain_fraction=1.0)
        mde = pymde.MDE(
            A.shape[0],
            dim,
            spg.edges.to("cpu"),
            pymde.losses.Quadratic(deviations=spg.distances.to("cpu")),
            pymde.constraints.Standardized(),
        )
        X = mde.embed().cpu().numpy()
        return X, node_labels

    # --- Node2Vec / STRND (PecanPy backend) ---
    if mode in {"STRND", "n2v"}:
        try:
            from pecanpy import pecanpy
        except Exception:
            raise ImportError(
                "PecanPy (node2vec backend) is required for STRND/n2v. "
                "Install with: pip install pecanpy"
            )

        # Build edge list as numpy array with optional weights
        src, dst, w, node_labels = _load_edge_list(edge_list)
        has_weights = not np.allclose(w, 1.0)

        # Initialize PecanPy
        backend = pecanpy.SparseOTF()  # most memory-efficient backend
        edges = np.vstack([src, dst]).T
        backend.read_edgelist(edges, weighted=has_weights)

        print(f"[edgerecon] Running {mode} embedding on {len(node_labels)} nodes...")

        # Learn embeddings (using default node2vec parameters)
        emb = backend.learn_embedding(dim=64)  # you can expose this as param
        emb = np.asarray(emb)

        # Ensure embeddings align with sorted node order
        if hasattr(backend, "node2id"):
            order = [backend.node2id[n] for n in sorted(backend.node2id, key=backend.node2id.get)]
            emb = emb[np.argsort(order)]

        # If requested dim < learned dim, reduce (e.g., UMAP)
        if emb.shape[1] > dim:
            try:
                import umap
                reducer = umap.UMAP(
                    n_components=dim, n_neighbors=n_neighbors, min_dist=1.0, random_state=random_state
                )
                emb = reducer.fit_transform(emb)
            except Exception:
                print("[edgerecon] UMAP not installed; returning full-d embeddings.")

        return emb, node_labels

    raise ValueError(f"Unknown mode '{mode}'.")


# -------------------------- Landmark Isomap helpers -------------------------- #

def _landmark_isomap_unweighted(A: sp.spmatrix, *, dim: int, n_landmarks: int, random_state: int) -> np.ndarray:
    """
    BFS-based geodesic distances to random landmarks (unweighted),
    MDS on landmarks + triangulation for the rest.
    """
    # Build adjacency list for BFS
    rows, cols = A.nonzero()
    import collections
    graph = collections.defaultdict(set)
    for i, j in zip(rows, cols):
        graph[i].add(j)
        graph[j].add(i)
    N = A.shape[0]
    L = min(n_landmarks, N)
    rng = np.random.default_rng(random_state)
    landmarks = rng.choice(N, L, replace=False)

    # BFS distances to landmarks
    from collections import deque
    all_to_L = np.full((N, L), np.inf)

    def bfs(source):
        dist = {node: np.inf for node in graph}
        dist[source] = 0
        q = deque([source])
        while q:
            u = q.popleft()
            for v in graph[u]:
                if dist[v] == np.inf:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    for j, l in enumerate(landmarks):
        d = bfs(l)
        for i, val in d.items():
            all_to_L[i, j] = val

    # MDS on landmarks (L x L), then triangulate others
    D_LL = all_to_L[landmarks, :]
    D_LL = _symmetrize(D_LL)

    mds = manifold.MDS(n_components=dim, metric=True, random_state=random_state, dissimilarity="precomputed")
    Lcoords = np.array(mds.fit_transform(D_LL))

    D2_LL = D_LL ** 2
    D2_all = all_to_L ** 2
    mean_col = D2_LL.mean(axis=0)
    L_pinv = np.linalg.pinv(Lcoords)
    X = (0.5 * (mean_col - D2_all)) @ L_pinv.T
    return X


def _landmark_isomap_from_distances(D: np.ndarray, *, dim: int, n_landmarks: int, random_state: int) -> np.ndarray:
    """Landmark Isomap when full weighted distances are available."""
    N = D.shape[0]
    L = min(n_landmarks, N)
    rng = np.random.default_rng(random_state)
    landmarks = rng.choice(N, L, replace=False)
    all_to_L = D[:, landmarks]
    D_LL = all_to_L[landmarks]
    D_LL = _symmetrize(D_LL)

    mds = manifold.MDS(n_components=dim, metric=True, random_state=random_state, dissimilarity="precomputed")
    Lcoords = np.array(mds.fit_transform(D_LL))

    D2_LL = D_LL ** 2
    D2_all = all_to_L ** 2
    mean_col = D2_LL.mean(axis=0)
    L_pinv = np.linalg.pinv(Lcoords)
    X = (0.5 * (mean_col - D2_all)) @ L_pinv.T
    return X


def _symmetrize(a: np.ndarray) -> np.ndarray:
    return a + a.T - np.diag(a.diagonal())
