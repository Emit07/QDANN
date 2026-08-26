"""Checks on the Eq. 2 fit, which every satellite feature downstream is built from."""

import numpy
import pytest

from qdann import harmonics


def _series(coefficients: numpy.ndarray, t: numpy.ndarray) -> numpy.ndarray:
    return harmonics.design(t) @ coefficients


def test_known_coefficients_are_recovered():
    t = numpy.linspace(0, 1, 20)
    coefficients = numpy.array([3.0, 0.5, -1.25, 0.75, 0.1, -0.3, 0.6])
    fitted = harmonics.fit(t=t, y=_series(coefficients, t))
    assert fitted == pytest.approx(coefficients, abs=1e-10)


def test_a_constant_series_is_all_offset():
    fitted = harmonics.fit(t=numpy.linspace(0, 1, 20), y=numpy.full(20, 2.5))
    assert fitted[0] == pytest.approx(2.5)
    assert fitted[1:] == pytest.approx(numpy.zeros(6), abs=1e-10)


def test_too_few_observations_are_not_fitted():
    # irregular, as a real cloud-masked Landsat record is
    t = numpy.array([0.05, 0.18, 0.31, 0.47, 0.62, 0.88])
    assert numpy.isnan(harmonics.fit(t=t, y=numpy.arange(6.0))).all()
    # one more observation reaches the rank floor
    t = numpy.append(t, 0.73)
    assert numpy.isfinite(harmonics.fit(t=t, y=numpy.arange(7.0))).all()


def test_evenly_spaced_dates_alias_below_the_nyquist_limit():
    """Eq. 2's fastest term is w*n = 4.5 cycles over the window, so a regular grid of
    fewer than 10 dates cannot determine 7 coefficients however many of them there are:
    the harmonics repeat on the grid. Landsat revisits irregularly once cloud masking has
    thinned the record, so this is the rank check earning its keep rather than a floor
    that could be raised."""
    for m in (7, 10):
        t = numpy.linspace(0, 1, m)
        assert numpy.linalg.matrix_rank(harmonics.design(t)) < 7
        assert numpy.isnan(harmonics.fit(t=t, y=numpy.arange(float(m)))).all()


def test_repeated_dates_are_not_fitted():
    """A county on a WRS-2 row seam is imaged twice on one day; the mosaic exists to
    collapse that, and an unmosaicked series is rank-deficient rather than wrong."""
    t = numpy.repeat(numpy.linspace(0, 1, 5), 2)
    assert numpy.isnan(harmonics.fit(t=t, y=numpy.arange(10.0))).all()


def test_missing_observations_are_dropped_not_propagated():
    t = numpy.linspace(0, 1, 20)
    coefficients = numpy.array([3.0, 0.5, -1.25, 0.75, 0.1, -0.3, 0.6])
    y = _series(coefficients, t)
    y[[2, 11]] = numpy.nan
    assert harmonics.fit(t=t, y=y) == pytest.approx(coefficients, abs=1e-10)


def test_design_matches_eq_2_term_by_term():
    t = numpy.array([0.0, 0.25, 0.6])
    a = harmonics.design(t)
    assert a.shape == (3, 7)
    assert a[:, 0] == pytest.approx(numpy.ones(3))
    for k in (1, 2, 3):
        angle = 2 * numpy.pi * harmonics.OMEGA * k * t
        assert a[:, 2 * k - 1] == pytest.approx(numpy.cos(angle))
        assert a[:, 2 * k] == pytest.approx(numpy.sin(angle))
