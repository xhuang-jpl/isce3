import numpy as np
from functools import lru_cache

def ncoeffs(degree: int) -> int:
    """Return the number of coefficients in a 2D polynomial
    with total degree ≤ `degree`."""
    return (degree + 1) * (degree + 2) // 2

def normalize(x, xmin, xmax):
    """Normalize values linearly to [0, 1]."""
    if xmax == xmin:
        return np.zeros_like(x, dtype=float)
    return (x - xmin) / (xmax - xmin)

def build_design_matrix(lines_n, pixels_n, degree: int):
    """
    Build the design matrix for a 2D polynomial regression.

    For each observation i:
        A[i, :] = [l^(j-k) * p^k] for all 0 <= j <= degree, 0 <= k <= j
    where l, p are the normalized line/pixel coordinates.
    """
    N = lines_n.size
    A = np.zeros((N, ncoeffs(degree)), dtype=float)
    idx = 0
    for j in range(degree + 1):
        for k in range(j + 1):
            A[:, idx] = (lines_n ** (j - k)) * (pixels_n ** k)
            idx += 1
    return A

def cholesky_solve(N_mat, rhs):
    """
    Solve a symmetric positive-definite linear system:
        N_mat * x = rhs
    using Cholesky decomposition.
    Returns:
        x : solution vector
        L : lower-triangular Cholesky factor of N_mat
    """
    L = np.linalg.cholesky(N_mat)
    y = np.linalg.solve(L, rhs)     # Solve L y = rhs
    x = np.linalg.solve(L.T, y)     # Solve L^T x = y
    return x, L

def invert_from_cholesky(L):
    """
    Compute the inverse of A given its Cholesky factorization A = L L^T.
    Returns:
        A_inv = (L^-1)^T * L^-1
    """
    Linv = np.linalg.inv(L)
    return Linv.T @ Linv

def polyfit_offsets(
    Data,
    degree=2,
    prf=None, abw=None, rsr=None, rbw=None,  # SAR parameters for sigma estimation
    sigmaL=None, sigmaP=None,                # Prior standard deviations
    crit_value=0.1,
    max_iterations=50
):
    """
    Robust polynomial fitting for SAR offset modeling with iterative outlier removal.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (N, 6), each row:
        [id, line, pixel, dL, dP, coherence].
    degree : int
        Degree of the 2D polynomial (1=affine, 2=quadratic, etc.).
    coherence_threshold : float
        Discard tie points with coherence below this threshold.
    prf, abw, rsr, rbw : float, optional
        SAR system parameters used to estimate SIGMAL/SIGMAP if provided.
    sigmaL, sigmaP : float, optional
        Prior standard deviations for line and pixel offsets.
        If not provided, they are estimated from SAR parameters.
    crit_value : float
        Critical value for the w-test (typically 3.0).
    max_iterations : int
        Maximum number of outlier rejection iterations.

    Returns
    -------
    dict
        {
          "coefL": np.ndarray, polynomial coefficients for line direction,
          "coefP": np.ndarray, polynomial coefficients for pixel direction,
          "inliers": np.ndarray, subset of input data kept after outlier removal,
          "removed_indices": list of int, indices of removed tie points,
          "degree": int, polynomial degree used,
          "design_nunk": int, number of unknown coefficients.
        }
    """
    # Estimate or set SIGMAL, SIGMAP
    if sigmaL is None or sigmaP is None:
        if (prf is None) or (abw is None) or (rsr is None) or (rbw is None):
            SIGMAL = 0.15 / 1.1
            SIGMAP = 0.10 / 1.1
        else:
            SIGMAL = 0.15 / (prf / abw)
            SIGMAP = 0.10 / (rsr / rbw)
    else:
        SIGMAL, SIGMAP = float(sigmaL), float(sigmaP)

    minL, maxL = Data[:,1].min(), Data[:,1].max()
    minP, maxP = Data[:,2].min(), Data[:,2].max()
    Nunk = ncoeffs(degree)
    removed_indices = []
    iteration = 0

    while True:
        lines = Data[:, 1]
        pixels = Data[:, 2]
        yL = Data[:, 3:4]
        yP = Data[:, 4:5]
        Nobs = Data.shape[0]

        if Nobs <= Nunk:
            break  # Not enough redundancy to continue

        Ln = normalize(lines, minL, maxL)
        Pn = normalize(pixels, minP, maxP)
        A = build_design_matrix(Ln, Pn, degree)

        At = A.T
        Nmat = At @ A
        rhsL = At @ yL
        rhsP = At @ yP

        try:
            xL, Lc = cholesky_solve(Nmat, rhsL)
            xP, _  = cholesky_solve(Nmat, rhsP)
        except np.linalg.LinAlgError:
            eps = 1e-8
            Nmat_reg = Nmat + eps * np.eye(Nmat.shape[0])
            xL, Lc = cholesky_solve(Nmat_reg, rhsL)
            xP, _  = cholesky_solve(Nmat_reg, rhsP)

        Qx_hat = invert_from_cholesky(Lc)

        yL_hat = A @ xL
        yP_hat = A @ xP
        eL = yL - yL_hat
        eP = yP - yP_hat

        Qy_hat = A @ Qx_hat @ A.T
        Qe = -Qy_hat
        Qe[np.diag_indices_from(Qe)] += 1.0
        diag_Qe = np.clip(np.diag(Qe), 1e-12, None)

        wL = eL[:,0] / (np.sqrt(diag_Qe) * SIGMAL)
        wP = eP[:,0] / (np.sqrt(diag_Qe) * SIGMAP)
        wsum = wL**2 + wP**2

        maxwL = np.max(np.abs(wL))
        maxwP = np.max(np.abs(wP))
        max_any = max(maxwL, maxwP)

        if (max_any <= crit_value) or (iteration >= max_iterations):
            coefL = xL[:,0]
            coefP = xP[:,0]
            return {
                "coefL": coefL,
                "coefP": coefP,
                "inliers": Data,
                "removed_indices": removed_indices,
                "degree": degree,
                "design_nunk": Nunk
            }

        worst_idx_local = int(np.argmax(wsum))
        removed_indices.append(int(Data[worst_idx_local, 0]))
        Data = np.delete(Data, worst_idx_local, axis=0)
        iteration += 1

def _poly_design(line, pixel, degree, minL, maxL, minP, maxP):
    """
    Build a single design vector for a given (line, pixel),
    matching the coefficient order used in training:
        for j = 0..degree:
          for k = 0..j:
            term = l^(j-k) * p^k
    where l, p are normalized to [0,1] using the same min/max.
    """
    # normalize
    if maxL == minL:
        l = 0.0
    else:
        l = (line - minL) / (maxL - minL)

    if maxP == minP:
        p = 0.0
    else:
        p = (pixel - minP) / (maxP - minP)

    row = []
    for j in range(degree + 1):
        for k in range(j + 1):
            row.append((l ** (j - k)) * (p ** k))
    return np.array(row, dtype=float)

def predict_offsets(line, pixel, coefL, coefP, degree, minL, maxL, minP, maxP):
    """
    Predict (ΔL, ΔP) at a single (line, pixel).

    Parameters
    ----------
    line, pixel : float
    coefL, coefP : 1D arrays of length ncoeffs(degree)
    degree : int
    minL, maxL, minP, maxP : floats used during fitting for normalization

    Returns
    -------
    dL, dP : floats
    """
    v = _poly_design(line, pixel, degree, minL, maxL, minP, maxP)
    dL = float(v @ coefL)
    dP = float(v @ coefP)
    return dL, dP

@lru_cache(maxsize=None)
def _expo_pairs(degree: int):
    # Returns arrays A,B so term_i = l**A[i] * p**B[i]
    A, B = [], []
    for j in range(degree + 1):
        k = np.arange(j + 1)
        A.append(j - k)
        B.append(k)
    return np.concatenate(A).astype(np.int64), np.concatenate(B).astype(np.int64)

def _poly_design_batch(lines, pixels, degree, minL, maxL, minP, maxP):
    """
    Build the full design matrix for arrays of (lines, pixels).
    Returns shape: (N, (degree+1)(degree+2)//2)
    """
    lines = np.asarray(lines, dtype=float)
    pixels = np.asarray(pixels, dtype=float)

    # Normalize (safe for zero ranges)
    l = 0.0 if maxL == minL else (lines - minL) / (maxL - minL)
    p = 0.0 if maxP == minP else (pixels - minP) / (maxP - minP)

    # Flatten (we’ll reshape later)
    l = np.ravel(l)
    p = np.ravel(p)

    Aidx, Bidx = _expo_pairs(degree)
    ar = np.arange(degree + 1, dtype=np.int64)

    # Precompute powers once per sample
    l_pows = np.power(l[:, None], ar[None, :])  # (N, D+1)
    p_pows = np.power(p[:, None], ar[None, :])  # (N, D+1)

    # Gather needed powers and multiply to get terms in training order
    return l_pows.take(Aidx, axis=1) * p_pows.take(Bidx, axis=1)

def predict_offsets_batch(lines, pixels, coefL, coefP, degree, minL, maxL, minP, maxP):
    """
    Vectorized prediction for arrays of lines/pixels (same shape).

    Returns
    -------
    dL, dP : arrays with the same shape as `lines`/`pixels`.
    """
    lines = np.asarray(lines)
    pixels = np.asarray(pixels)
    out_shape = np.broadcast_shapes(lines.shape, pixels.shape)

    # Build design matrix in one vectorized pass
    A = _poly_design_batch(
        np.broadcast_to(lines, out_shape),
        np.broadcast_to(pixels, out_shape),
        degree, minL, maxL, minP, maxP
    )

    # One BLAS call for both outputs
    C = np.column_stack((np.asarray(coefL, dtype=float).ravel(),
                         np.asarray(coefP, dtype=float).ravel()))  # (nunk, 2)
    Y = A @ C  # (N, 2)

    dL = Y[:, 0].reshape(out_shape)
    dP = Y[:, 1].reshape(out_shape)
    return dL, dP

def test():

    rng = np.random.default_rng(0)
    N = 200
    lines = rng.uniform(0, 4000, size=N)
    pixels = rng.uniform(0, 6000, size=N)

    def true_model(l, p):
        return 0.1 + 2e-4*l + 3e-4*p + 1e-8*l*p + 1e-8*l*l + 1e-8*p*p

    dL_true = true_model(lines, pixels)
    dP_true = -true_model(lines, pixels) * 0.8

    noiseL = rng.normal(0, 0.05, size=N)
    noiseP = rng.normal(0, 0.05, size=N)

    dL = dL_true #+ noiseL
    dP = dP_true #+ noiseP
    coherence = rng.uniform(0.3, 1.0, size=N)

    outlier_idx = rng.choice(N, size=10, replace=False)
    print(outlier_idx)
    print(dL[outlier_idx])
    print(rng.normal(2.0, 0.5, size=10))

    dL[outlier_idx] += rng.normal(15.0, 5, size=10)
    dP[outlier_idx] += rng.normal(-15.0, 5, size=10)
    print(dL[outlier_idx])
    data = np.column_stack([
        np.arange(N), lines, pixels, dL, dP, coherence
    ])

    res = polyfit_offsets(
        data,
        degree=2,
        sigmaL=0.15/1.1,
        sigmaP=0.10/1.1,
        crit_value=0.001,
        max_iterations=200
    )

    print("Degree:", res["degree"])
    print("Nunk:", res["design_nunk"])
    print("CoefL:", res["coefL"])
    print("CoefP:", res["coefP"])
    print("Inliers:", res["inliers"].shape[0], "/", N)
    print("Removed IDs:", res["removed_indices"])

    # pull fit + normalization bounds (use the same min/max from the inliers you fitted)
    coefL = res["coefL"]
    coefP = res["coefP"]
    degree = res["degree"]

    inliers = res["inliers"]
    minL, maxL = inliers[:,1].min(), inliers[:,1].max()
    minP, maxP = inliers[:,2].min(), inliers[:,2].max()

    # single point prediction
    line0, pixel0 = 1234.5, 4567.8
    dL0, dP0 = predict_offsets(line0, pixel0, coefL, coefP, degree, minL, maxL, minP, maxP)
    print(dL0, dP0)

    # grid prediction (e.g., to visualize a surface)
    Lvec = np.linspace(minL, maxL, 50)
    Pvec = np.linspace(minP, maxP, 50)
    LL, PP = np.meshgrid(Lvec, Pvec, indexing="ij")
    dL_grid, dP_grid = predict_offsets_batch(LL, PP, coefL, coefP, degree, minL, maxL, minP, maxP)

    print(dL_grid, dP_grid)

if __name__ == "__main__":
    test()