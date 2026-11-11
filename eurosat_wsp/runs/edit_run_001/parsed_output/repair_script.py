"""EDIT experiment, variant A (gaussian_noise on layer3.0.conv1.weight) -- repair function.

Author: Claude Fable 5.1, run edit_run_001, 2026-11-10.

Strategy (tensor-only, data-free, no access to the original checkpoint or the
modified-index file):

1. Detect the damaged output filters.  Each output filter is a row of the
   [256, 1152] matrix.  Gaussian noise added to a row inflates its standard
   deviation; the robust rule  row_std > 3 * median(row_std)  separates the
   damaged rows from the healthy ones (in this checkpoint the healthy rows have
   std 0.025-0.040 and the damaged rows 0.169-0.182, a 4x gap).

2. Estimate the noise level from the tensor itself:
       sigma_noise^2 = var(damaged rows) - var(healthy rows).

3. Debiased PCA-Wiener denoising of the damaged rows using the healthy rows as
   the empirical prior.  The healthy rows are centred on their mean filter mu,
   their principal directions v_i and per-direction variances lambda_i are
   computed by SVD, and every lambda_i is debiased by the Marchenko-Pastur bulk
   edge  lambda_edge = median(lambda) * (1 + sqrt(p/n))^2  (p = 1152, n = 224),
   so that sampling-inflated eigenvalues of an isotropic bulk are not mistaken
   for structure.  A damaged row w is then estimated as
       w_hat = mu + sum_i g_i <w - mu, v_i> v_i ,
       g_i   = max(lambda_i - lambda_edge, 0) / (max(lambda_i - lambda_edge, 0) + sigma_noise^2).
   This is the MMSE (Wiener) estimator under the Gaussian prior
   w ~ N(mu, sum_i lambda_i v_i v_i^T) and white noise N(0, sigma_noise^2 I).

   For this layer the noise variance is ~35x the per-weight signal variance and
   no principal direction of the healthy rows exceeds the Marchenko-Pastur edge,
   so every gain g_i is 0 and the estimator reduces to its prior mean: each
   damaged filter is replaced by the mean healthy filter mu.  The function keeps
   the general form so that it would exploit low-rank structure if it existed.

4. Healthy rows are returned untouched; dtype/shape of the input are preserved.
"""

import torch


def repair_layer3_conv1(corrupted_tensor: torch.Tensor) -> torch.Tensor:
    """
    Args:
        corrupted_tensor: torch.Tensor of shape [256, 128, 3, 3] (float32)
    Returns:
        repaired_tensor: torch.Tensor of shape [256, 128, 3, 3] (float32)
    """
    W = corrupted_tensor.detach().clone()
    orig_dtype = W.dtype
    out_ch = W.shape[0]
    M = W.reshape(out_ch, -1).to(torch.float64)          # [256, 1152]
    n_rows, p = M.shape

    # --- 1. detect damaged output filters (robust per-row std rule) -------------
    row_std = M.std(dim=1)
    bad = row_std > 3.0 * row_std.median()
    if bad.sum() == 0 or bad.sum() == n_rows:
        return W                                            # nothing to repair

    clean = ~bad
    Mc = M[clean]                                           # healthy rows (prior sample)
    Mb = M[bad]                                             # damaged rows

    # --- 2. noise level estimated from the tensor itself ----------------------
    sigma_w2 = Mc.var().item()
    sigma_n2 = max(Mb.var().item() - sigma_w2, 1e-12)

    # --- 3. debiased PCA-Wiener (MMSE) estimate of the damaged rows ------------
    mu = Mc.mean(dim=0)                                     # mean healthy filter
    Xc = Mc - mu
    n_clean = Xc.shape[0]
    _, S, Vt = torch.linalg.svd(Xc, full_matrices=False)    # Vt: [k, 1152]
    lam = S ** 2 / n_clean                                  # per-direction variance
    lam_edge = lam.median() * (1.0 + (p / n_clean) ** 0.5) ** 2   # Marchenko-Pastur edge
    lam_db = (lam - lam_edge).clamp(min=0.0)
    gains = lam_db / (lam_db + sigma_n2)                    # Wiener gain per direction

    coef = (Mb - mu) @ Vt.T                                 # projections of damaged rows
    repaired_rows = mu + (coef * gains) @ Vt                # shrink in the eigenbasis

    M[bad] = repaired_rows

    repaired = M.to(orig_dtype).reshape_as(W)
    return repaired.contiguous()
