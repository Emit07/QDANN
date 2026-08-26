"""Harmonic regression over a GCVI time series.

Ma et al., Remote Sensing of Environment 315 (2024) 114427, Equation 2. See SPEC.md
section 1. The seven coefficients returned here are the satellite half of the feature
vector, and are computed identically for a county-year and for a pixel-year: the two
differ only in the spatial support the series was reduced over.
"""

import numpy

N_HARMONICS = 3
# Section 2.2, following Deines et al. 2021
OMEGA = 1.5


def design(t: numpy.ndarray, n: int = N_HARMONICS, w: float = OMEGA) -> numpy.ndarray:
    """Eq. 2 evaluated as a (len(t), 2n+1) matrix of [1, cos_k, sin_k] columns.

    `t` is the observation date normalized to [0, 1] over the crop's window, which for
    winter wheat straddles two calendar years.
    """
    t = numpy.asarray(t, dtype=float)
    angles = 2 * numpy.pi * w * numpy.outer(t, numpy.arange(1, n + 1))
    ret = numpy.empty((len(t), 2 * n + 1))
    ret[:, 0] = 1.0
    ret[:, 1::2] = numpy.cos(angles)
    ret[:, 2::2] = numpy.sin(angles)
    return ret


def fit(
    t: numpy.ndarray,
    y: numpy.ndarray,
    n: int = N_HARMONICS,
    w: float = OMEGA,
    min_observations: int | None = None,
) -> numpy.ndarray:
    """Least-squares fit of Eq. 2, returning [c, a_1, b_1, ... a_n, b_n].

    Answers with NaNs rather than a fit that cannot be trusted. `min_observations`
    defaults to the number of coefficients, which is a rank floor and not a quality one:
    at exactly 2n+1 distinct dates the fit interpolates with zero residual. Section 2.2
    reports around 20 cloud-free scenes per cropping year, dropping to 15 in 2012 when
    only Landsat 7 flew, so a county-year near the floor is worth looking at.
    """
    t = numpy.asarray(t, dtype=float)
    y = numpy.asarray(y, dtype=float)
    n_coefficients = 2 * n + 1
    if min_observations is None:
        min_observations = n_coefficients
    unfitted = numpy.full(n_coefficients, numpy.nan)
    # a county-scene with no unmasked pixel reduces to a missing band rather than an error
    usable = numpy.isfinite(t) & numpy.isfinite(y)
    if int(usable.sum()) < min_observations:
        return unfitted
    coefficients, _, rank, _ = numpy.linalg.lstsq(
        design(t=t[usable], n=n, w=w), y[usable], rcond=None
    )
    # rank-deficiency survives the count check: repeated dates leave duplicate rows, and
    # a regular grid of fewer than 2*w*n+1 dates aliases the fastest harmonic onto a slower
    # one. lstsq answers either with a minimum-norm solution rather than an error
    return coefficients if rank == n_coefficients else unfitted
