"""Tensor-only repair methods for a damaged conv layer (EDIT experiment).

Every method has the signature ``repair(W: Tensor[out, in, kh, kw]) -> Tensor`` and
uses nothing but the corrupted tensor itself: no data, no gradients, no original
checkpoint, no index files.

Damaged output filters are found from the tensor's own statistics
(``detect_damaged_rows``): a filter is damaged when its standard deviation is far
above the layer median (additive noise) or when most of its weights are exactly
zero (pruning, filter removal).

The methods, roughly from naive to principled:

* ``zero_rows``        switch damaged filters off.
* ``norm_match``       rescale damaged filters to the median healthy norm.
* ``clip_range``       clip damaged filters to the healthy 99.9th-percentile range.
* ``global_svd``       truncated SVD of the whole [out, in*kh*kw] matrix (rank 64),
                       the textbook "low-rank denoising" recipe; no detection.
* ``mean_filter``      replace damaged filters with the mean healthy filter.
* ``conditional_mean`` MMSE estimate under a Gaussian prior fitted on the healthy
                       filters. With additive noise it is a shrunken Wiener filter,
                       with pruning it imputes the zeroed weights from the surviving
                       ones, and with fully removed filters it equals ``mean_filter``.
"""

from __future__ import annotations

from typing import Callable

import torch

Repair = Callable[[torch.Tensor], torch.Tensor]


# --------------------------------------------------------------------- helpers


def _as_matrix(W: torch.Tensor) -> torch.Tensor:
    return W.detach().reshape(W.shape[0], -1).to(torch.float64).clone()


def _back(M: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    return M.to(W.dtype).reshape(W.shape).contiguous()


def detect_damaged_rows(M: torch.Tensor, std_factor: float = 3.0,
                        zero_frac: float = 0.5) -> torch.Tensor:
    """Boolean mask of damaged rows (output filters) of an [out, fan_in] matrix."""
    row_std = M.std(dim=1)
    noisy = row_std > std_factor * row_std.median()
    sparse = (M == 0).double().mean(dim=1) > zero_frac
    return noisy | sparse


def _split(W: torch.Tensor):
    M = _as_matrix(W)
    bad = detect_damaged_rows(M)
    return M, bad, ~bad


# --------------------------------------------------------------------- methods


def identity(W: torch.Tensor) -> torch.Tensor:
    """No repair (the corrupted model)."""
    return W.clone()


def zero_rows(W: torch.Tensor) -> torch.Tensor:
    M, bad, _ = _split(W)
    M[bad] = 0.0
    return _back(M, W)


def norm_match(W: torch.Tensor) -> torch.Tensor:
    M, bad, good = _split(W)
    target = M[good].norm(dim=1).median()
    norms = M[bad].norm(dim=1, keepdim=True).clamp_min(1e-12)
    M[bad] = M[bad] / norms * target
    return _back(M, W)


def clip_range(W: torch.Tensor) -> torch.Tensor:
    M, bad, good = _split(W)
    healthy = M[good].abs().flatten()
    k = max(1, int(0.999 * healthy.numel()))
    limit = healthy.kthvalue(k).values
    M[bad] = M[bad].clamp(-limit, limit)
    return _back(M, W)


def global_svd(W: torch.Tensor, rank: int = 64) -> torch.Tensor:
    M = _as_matrix(W)
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    M = (U[:, :rank] * S[:rank]) @ Vh[:rank]
    return _back(M, W)


def mean_filter(W: torch.Tensor) -> torch.Tensor:
    M, bad, good = _split(W)
    M[bad] = M[good].mean(dim=0)
    return _back(M, W)


def conditional_mean(W: torch.Tensor) -> torch.Tensor:
    """MMSE repair under a shrunken Gaussian prior N(mu, S + lam*I) from healthy rows.

    * Additive noise: observed row y = w + n, n ~ N(0, s2*I) with s2 estimated as
      var(damaged) - var(healthy). Eigen-directions are debiased by the
      Marchenko-Pastur edge, so a flat (structureless) spectrum gives all-zero
      gains and the estimate is the prior mean.
    * Pruned rows: exact zeros are treated as missing, non-zeros as exact
      observations; missing entries get E[w_m | w_o] = mu_m + S_mo (S_oo+lam I)^-1 (w_o - mu_o).
    * Fully zeroed rows have nothing observed and fall back to mu.
    """
    M, bad, good = _split(W)
    H = M[good]
    mu = H.mean(dim=0)
    X = H - mu
    n, p = X.shape

    noisy = bad & ((M == 0).double().mean(dim=1) <= 0.5)
    sparse = bad & ~noisy

    if noisy.any():
        _, S, Vh = torch.linalg.svd(X, full_matrices=False)
        lam = S ** 2 / n
        edge = lam.median() * (1.0 + (p / n) ** 0.5) ** 2
        lam = (lam - edge).clamp_min(0.0)
        s2 = max(M[noisy].var().item() - H.var().item(), 1e-12)
        gains = lam / (lam + s2)
        coef = (M[noisy] - mu) @ Vh.T
        M[noisy] = mu + (coef * gains) @ Vh

    if sparse.any():
        cov = X.T @ X / n
        ridge = cov.diagonal().mean()            # shrinkage toward isotropy (n << p)
        for r in torch.nonzero(sparse).flatten().tolist():
            row = M[r]
            obs = row != 0
            if not obs.any():
                M[r] = mu
                continue
            miss = ~obs
            C_oo = cov[obs][:, obs] + ridge * torch.eye(int(obs.sum()), dtype=cov.dtype)
            C_mo = cov[miss][:, obs]
            delta = torch.linalg.solve(C_oo, row[obs] - mu[obs])
            filled = row.clone()
            filled[miss] = mu[miss] + C_mo @ delta
            M[r] = filled
    return _back(M, W)


METHODS: dict[str, Repair] = {
    "corrupted": identity,
    "zero_rows": zero_rows,
    "norm_match": norm_match,
    "clip_range": clip_range,
    "global_svd": global_svd,
    "mean_filter": mean_filter,
    "conditional_mean": conditional_mean,
}

LABELS: dict[str, str] = {
    "corrupted": "No repair",
    "zero_rows": "Zero damaged filters",
    "norm_match": "Rescale to healthy norm",
    "clip_range": "Clip to healthy range",
    "global_svd": "Global truncated SVD (rank 64)",
    "mean_filter": "Mean healthy filter",
    "conditional_mean": "Conditional mean (Gaussian prior)",
}
