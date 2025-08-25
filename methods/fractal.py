"""
Description
-----------
This module implements the Higuchi Fractal Dimension (HFD) feature extractor for EEG signals.
It is intended to be used as an 'fe' (feature extraction) component within the
pos_folding stage of the bciflow kfold pipeline so that feature computation occurs
inside each cross-validation fold, preventing information leakage.

The Higuchi Fractal Dimension estimates the complexity of a time series by evaluating
its length across multiple temporal resolutions (k values) and performing a log–log
linear fit. Higher HFD values generally indicate greater signal irregularity.

Function
--------
The public function higuchi_fractal receives an eegdata dictionary containing the key 'X'
with shape (trials, bands, electrodes, time) or (trials, electrodes, time) or (trials, 1, electrodes, time)
and returns the same dictionary with 'X' replaced by the HFD feature values.

Parameters (function)
---------------------
eegdata : dict
    EEG data dictionary. Must contain key 'X'.
kmax : int, optional
    Maximum k evaluated in the Higuchi algorithm (default = 100).
flating : bool, optional
    If True returns (trials, features); else (trials, bands, electrodes).
**kwargs : dict
    Accepts legacy synonym 'flattening' (boolean). If provided it is OR'ed with flating.

Returns
-------
dict
    The same dictionary with 'X' replaced by HFD feature values.

Notes
-----
- Supports both parameter names flating (preferred) and flattening (legacy).
- Uses log-spaced k values for efficiency; each signal is mean-centered.
- Stateless transform suitable for bciflow pos_folding under key 'fe'.

Reference
---------
Higuchi, T. (1988). Physica D: Nonlinear Phenomena, 31(2), 277–283.
"""

import numpy as np


def higuchi_fractal(
    eegdata: dict,
    kmax: int = 100,
    flating: bool = False,
    **kwargs,
) -> dict:
    """
    Computes the Higuchi Fractal Dimension (HFD) per channel/component.

    Stateless feature extraction (no fit phase). Intended for bciflow pos_folding under key 'fe'.

    Parameters
    ----------
    eegdata : dict
        Dictionary with key 'X'. Accepted shapes:
            (trials, bands, electrodes, time) or (trials, electrodes, time).
    kmax : int, optional
        Maximum k (default = 100). Must be >= 2.
    flating : bool, optional
        If True output becomes (trials, features); otherwise (trials, bands, electrodes).
    **kwargs : dict
        Optional legacy synonym 'flattening' (bool). If provided it is combined with flating.

    Returns
    -------
    output : dict
        Same dictionary with 'X' replaced by HFD values.

    Raises
    ------
    ValueError
        If eegdata has invalid type, missing 'X', kmax < 2, or X has unsupported shape.
    """
    if not isinstance(eegdata, dict):
        raise ValueError("eegdata must be a dict containing key 'X'.")
    if "X" not in eegdata:
        raise KeyError("eegdata must contain key 'X'.")
    if not isinstance(kmax, int):
        raise ValueError("kmax must be an integer >= 2.")
    if kmax < 2:
        kmax = 2

    if "flattening" in kwargs:
        flating = flating or bool(kwargs["flattening"])
    if "flating" in kwargs:
        flating = flating or bool(kwargs["flating"])

    X = eegdata["X"]
    if not isinstance(X, np.ndarray):
        raise ValueError("eegdata['X'] must be a numpy.ndarray.")

    if X.ndim == 3:
        X = X[:, np.newaxis, :, :]
    elif X.ndim != 4:
        raise ValueError(
            f"Unsupported shape for higuchi_fractal: {X.shape}. Expected 3D (trials, electrodes, time) or 4D (trials, bands, electrodes, time)."
        )

    trials, bands, electrodes, time = X.shape
    X_flat = X.reshape(trials * bands * electrodes, time)

    n_time = time
    if n_time < 10:
        hfd_vals = np.zeros(X_flat.shape[0], dtype=float)
    else:
        max_k = min(kmax, n_time // 2)
        if max_k < 2:
            hfd_vals = np.zeros(X_flat.shape[0], dtype=float)
        else:
            scales = np.unique(np.logspace(0, np.log10(max_k), num=10, dtype=int))
            scales = scales[scales >= 1]
            hfd_vals = np.empty(X_flat.shape[0], dtype=float)
            for i, sig in enumerate(X_flat):
                sig = sig - sig.mean()
                lk = np.zeros(scales.shape[0], dtype=float)
                for si, k in enumerate(scales):
                    sum_l = 0.0
                    count = 0
                    for m in range(k):
                        idx = np.arange(m, n_time, k)
                        if idx.size >= 2:
                            seg = sig[idx]
                            d = np.abs(np.diff(seg))
                            L_mk = (d.sum() * (n_time - 1)) / (d.size * k)
                            sum_l += L_mk
                            count += 1
                    if count == 0:
                        lk[si] = 0.0
                    else:
                        Lk = sum_l / count
                        lk[si] = np.log(Lk + 1e-12)
                valid = ~np.isinf(lk) & ~np.isnan(lk)
                if valid.sum() < 2:
                    hfd_vals[i] = 0.0
                else:
                    hfd_vals[i] = float(
                        np.polyfit(np.log(1.0 / scales[valid]), lk[valid], 1)[0]
                    )

    if flating:
        eegdata["X"] = hfd_vals.reshape(trials, bands * electrodes)
    else:
        eegdata["X"] = hfd_vals.reshape(trials, bands, electrodes)
    return eegdata
