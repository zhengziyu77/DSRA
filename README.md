# Empowering Heterogeneous Graph Foundation Models via Decoupled Relation Alignment

This is the official PyTorch implementation for the paper: **"Empowering Graph Foundation Models for Heterogeneous Graphs via Decoupled Relational Alignment"**.

## 📖 Introduction

**DRSA** is a **plug-and-play, meta-path-free** alignment framework. It fundamentally shifts the paradigm by decoupling feature semantics from relational structures. By projecting node features into a dual-relational subspace and utilizing a feature-structure decoupled representation, DRSA provides a unified, well-calibrated latent space for downstream heterogeneous graph foundation models.



## 🚀 Quick Start

Here is a complete example of how to align a heterogeneous graph using DRSA. We provide a utility function to convert PyTorch Geometric (PyG) `edge_index_dict` directly into the required format.

```python
import torch
import numpy as np

# 1. Create dummy heterogeneous graph data (e.g., Academic Network)
num_authors = 100
num_papers = 200
dim_author = 64
dim_paper = 128

# Node features (numpy arrays)
X_dict = {
    'author': np.random.randn(num_authors, dim_author),
    'paper': np.random.randn(num_papers, dim_paper)
}

# Edge indices in PyG format (PyTorch tensors)
# e.g., author (0-99) writes paper (0-199)
edge_index_dict = {
    ('author', 'writes', 'paper'): torch.randint(0, num_authors, (2, 500)),
    ('paper', 'cited_by', 'paper'): torch.randint(0, num_papers, (2, 800))
}

# 2. Convert PyG edge_index to Normalized SciPy sparse matrices
R_dict, _ = edge_index_dict_to_R_dict(edge_index_dict, normalize=True)

# 3. Initialize and Fit the DRSA Model
model = DRSA(
    k=50,          # Target aligned dimension
    rank_m=32,     # Low-rank projection dimension
    gamma=0.1,     # Feature regularization
    beta=0.1,      # Structural residual penalty
    n_iter=30      # Optimization iterations
)

print("Starting DRSA Alignment...")
model.fit(X_dict, R_dict)

# 4. Extract Aligned Embeddings
H_author = model.transform('author')
H_paper = model.transform('paper')

print(f"Aligned Author Shape: {H_author.shape}") # Expected: (100, 50)
print(f"Aligned Paper Shape: {H_paper.shape}")   # Expected: (200, 50)

# Now you can feed H_dict into any Graph Foundation Model!
```

---

## ⚙️ Parameter Details

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `k` | `int` | `50` | The unified embedding dimension for the aligned latent space. |
| `rank_m` | `int` | `32` | The rank for the dual-relational low-rank projection matrices ($A$ and $B$). |
| `gamma` | `float` | `0.1` | Regularization coefficient ($\gamma$) for the semantic projection matrix $P$. |
| `beta` | `float` | `0.1` | Penalty coefficient ($\beta$) for the structural residual term $E$. |
| `n_iter` | `int` | `50` | Number of alternating minimization (BCD) iterations. |

### Optimization Process
The `fit` function iteratively updates the aligned representations in two phases:
1. **Structure-Driven Estimation:** Updates the unified latent feature $H$ using the relational topology and dual-relational operators via Alternating Least Squares.
2. **Feature Decomposition:** Decomposes the updated target into a semantic projection ($X \cdot P$) and a structural residual ($E$) using Ridge Regression.



