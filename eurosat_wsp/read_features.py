"""Weight-space fingerprints used in the READ experiment.

All functions take plain tensors/arrays, so they work on checkpoints, on weights
parsed from a prompt, or on any SmallCNN-like model:

* ``conv1``      : array [16, 3, 3, 3]  (out, RGB, kh, kw)
* ``classifier`` : array [10, 128]      (classes, features)

The two statistics that separated the augmentation regimes in run read_run_001:

* ``luminance_fraction(conv1)``  share of each filter's energy carried by its
  channel-mean kernel. Colour jitter and random grayscale push R, G and B kernels
  toward a common luminance kernel, so it rises with augmentation strength.
* ``effective_rank(classifier)`` exp(entropy) of the normalised singular values of
  the head. Heavy augmentation concentrates the head on fewer robust features, so
  it falls with augmentation strength.

``seed_groups`` recovers shared initialisations from pairwise conv1 correlation.
"""

from __future__ import annotations

import numpy as np


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def luminance_fraction(conv1) -> float:
    W = _np(conv1)
    K = W.reshape(W.shape[0], W.shape[1], -1)                   # [out, C, k*k]
    mean_k = K.mean(axis=1, keepdims=True)
    lum = (mean_k ** 2).sum(axis=(1, 2)) * K.shape[1]
    total = (K ** 2).sum(axis=(1, 2)) + 1e-12
    return float((lum / total).mean())


def channel_cosine(conv1) -> float:
    W = _np(conv1)
    K = W.reshape(W.shape[0], W.shape[1], -1)
    K = K / (np.linalg.norm(K, axis=2, keepdims=True) + 1e-12)
    pairs = [(0, 1), (0, 2), (1, 2)]
    return float(np.mean([(K[:, a] * K[:, b]).sum(axis=1).mean() for a, b in pairs]))


def singular_values(classifier) -> np.ndarray:
    return np.linalg.svd(_np(classifier), compute_uv=False)


def effective_rank(classifier) -> float:
    s = singular_values(classifier)
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def condition_ratio(classifier) -> float:
    s = singular_values(classifier)
    return float(s[0] / s[-1])


def row_norm_cv(classifier) -> float:
    r = np.linalg.norm(_np(classifier), axis=1)
    return float(r.std() / r.mean())


def correlation_matrix(tensors) -> np.ndarray:
    """Pearson correlation between flattened tensors (one per model)."""
    return np.corrcoef(np.stack([_np(t).ravel() for t in tensors]))


def seed_groups(conv1_list, threshold: float = 0.5) -> list[list[int]]:
    """Connected components of the 'conv1 correlation > threshold' graph."""
    C = correlation_matrix(conv1_list)
    n = len(conv1_list)
    unseen, groups = set(range(n)), []
    while unseen:
        stack, comp = [unseen.pop()], []
        while stack:
            i = stack.pop()
            comp.append(i)
            for j in list(unseen):
                if C[i, j] > threshold:
                    unseen.remove(j)
                    stack.append(j)
        groups.append(sorted(comp))
    return sorted(groups)


FEATURES = {
    "luminance_fraction": lambda m: luminance_fraction(m["conv1.weight"]),
    "channel_cosine": lambda m: channel_cosine(m["conv1.weight"]),
    "head_effective_rank": lambda m: effective_rank(m["classifier.2.weight"]),
    "head_condition_ratio": lambda m: condition_ratio(m["classifier.2.weight"]),
    "head_row_norm_cv": lambda m: row_norm_cv(m["classifier.2.weight"]),
}


def fingerprint(state_dict) -> dict[str, float]:
    """All READ features for one SmallCNN state_dict."""
    return {name: fn(state_dict) for name, fn in FEATURES.items()}
