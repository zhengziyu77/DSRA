
import torch
import numpy as np
import random
from scipy.sparse import csr_matrix
from typing import Dict, Optional, Tuple
class DRSA:

    def __init__(
            self,
            k=50,
            rank_m=32,
            gamma=0.1,
            beta=0.1,
            n_iter=50,
            random_state=42,
    ):
        self.k = k
        self.rank_m = rank_m
        self.gamma = gamma
        self.beta = beta
        self.n_iter = n_iter
        self.random_state = random_state

        self.H_dict = {}
        self.P_dict = {}
        self.E_dict = {}

        self.A_dict = {}
        self.B_dict = {}
        self.M_dict = {}


        self.has_feat = {}
        self._X_pinv_cache = {}

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

    def fit(self, X_dict: Dict[str, np.ndarray],
            R_dict: Dict[str, Dict[str, csr_matrix]]):

        np.random.seed(self.random_state)
        random.seed(self.random_state)

        node_types = list(X_dict.keys())


        for t in node_types:
            n, d = X_dict[t].shape

            if d > 1:
                self.has_feat[t] = True
                self.P_dict[t] = np.random.randn(d, self.k) * 0.01

                self.E_dict[t] = np.random.randn(n, self.k) * 0.01
                self.H_dict[t] = X_dict[t] @ self.P_dict[t] + self.E_dict[t]

                XtX = X_dict[t].T @ X_dict[t]
                self._X_pinv_cache[t] = np.linalg.inv(
                    XtX + self.gamma * np.eye(d)
                ) @ X_dict[t].T

                self.A_dict[t] = np.random.randn(self.k, self.rank_m) * 0.1
                self.B_dict[t] = np.random.randn(self.k, self.rank_m) * 0.1


            else:
                self.has_feat[t] = False
                self.E_dict[t] = np.random.randn(n, self.k) * 0.01
                self.H_dict[t] = self.E_dict[t]

        for it in range(self.n_iter):

            for t in node_types:
                k = self.k
                n = self.H_dict[t].shape[0]

                coeff = self.beta * np.eye(k)
                rhs = np.zeros((k, n))

                for t2 in node_types:

                    # --- t -> t2 ---
                    if t in R_dict and t2 in R_dict[t]:
                        R = R_dict[t][t2]
                        if R.nnz == 0: continue

                        H2 = self.H_dict[t2]

                        A = self.A_dict[t]
                        B = self.B_dict[t2]

                        M = A @ B.T

                        HtH = H2.T @ H2
                        coeff +=  (M @ HtH @ M.T)
                        rhs +=  (R @ H2 @ M.T).T

                    # --- t2 -> t ---
                    if t2 in R_dict and t in R_dict[t2]:
                        R = R_dict[t2][t]
                        if R.nnz == 0: continue

                        H2 = self.H_dict[t2]

                        A = self.A_dict[t2]
                        B = self.B_dict[t]

                        M = A @ B.T

                        HtH = H2.T @ H2
                        coeff += (M.T @ HtH @ M)
                        rhs += (R.T @ H2 @ M).T

                coeff += 1e-6 * np.eye(k)

                try:
                    H_target = np.linalg.solve(coeff, rhs).T
                except:
                    H_target = np.linalg.lstsq(coeff, rhs, rcond=None)[0].T

                if self.has_feat[t]:
                    P = self._X_pinv_cache[t] @ (H_target - self.E_dict[t])
                    X_proj = X_dict[t] @ P
                    E = (H_target - X_proj) / (1 + self.beta)

                    self.P_dict[t] = P
                    self.E_dict[t] = E
                    self.H_dict[t] = X_proj + E
                else:
                    self.E_dict[t] = H_target / (1 + self.beta)
                    self.H_dict[t] = self.E_dict[t]

            if (it + 1) % 10 == 0:
                print(f"Iter {it + 1}/{self.n_iter} done")

        return self

    def transform(self, t):
        return self.H_dict[t]

import scipy.sparse as sp

def edge_index_dict_to_R_dict(
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        num_nodes_dict: Optional[Dict[str, int]] = None,
        normalize: bool = True
) -> Dict[str, Dict[str, csr_matrix]]:
    if num_nodes_dict is None:
        num_nodes_dict = {}
        for (src_type, _, dst_type), edge_index in edge_index_dict.items():
            if edge_index.shape[1] > 0:
                src_max = edge_index[0].max().item() + 1
                dst_max = edge_index[1].max().item() + 1
                num_nodes_dict[src_type] = max(num_nodes_dict.get(src_type, 0), src_max)
                num_nodes_dict[dst_type] = max(num_nodes_dict.get(dst_type, 0), dst_max)

    R_dict = {}

    for (src_type, edge_type, dst_type), edge_index in edge_index_dict.items():
        if edge_index is None or edge_index.shape[1] == 0: continue
        num_src = num_nodes_dict.get(src_type, 0)
        num_dst = num_nodes_dict.get(dst_type, 0)

        edge_index_np = edge_index.cpu().numpy()
        row, col = edge_index_np[0], edge_index_np[1]
        data = np.ones(len(row), dtype=np.float32)

        adj_matrix = csr_matrix((data, (row, col)), shape=(num_src, num_dst))


        if normalize:
            rowsum = np.array(adj_matrix.sum(axis=1)).flatten()
            colsum = np.array(adj_matrix.sum(axis=0)).flatten()
            rowsum[rowsum == 0] = 1.0
            colsum[colsum == 0] = 1.0

            d_mat = sp.diags(np.power(rowsum, -0.5))
            c_mat = sp.diags(np.power(colsum, -0.5))
            adj_matrix = d_mat.dot(adj_matrix).dot(c_mat)

        if src_type not in R_dict: R_dict[src_type] = {}
        R_dict[src_type][dst_type] = adj_matrix

    return R_dict




