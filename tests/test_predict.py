"""Stage 1's per-pixel features and the 5' products built from pixel yields."""

import numpy
import pandas
import pyproj
from rasterio.transform import Affine

from qdann import export, gee, harmonics, mpc, predict, target, weather


def _grid(lon: float, lat: float, n: int) -> mpc.Grid:
    crs = pyproj.CRS.from_epsg(32615)
    x, y = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    return mpc.Grid(crs=crs, transform=Affine(30, 0, x, 0, -30, y), width=n, height=n)


def _pixels(lon: float, lat: float, n: int, dates=numpy.array(["2018-06-01"])) -> mpc.Pixels:
    shape = (n * n, len(dates))
    return mpc.Pixels(
        on=_grid(lon, lat, n),
        crop=numpy.ones((n, n), dtype=bool),
        dates=dates,
        green=numpy.zeros(shape, numpy.float32),
        nir=numpy.zeros(shape, numpy.float32),
    )


def test_pixel_harmonics_recover_eq_2_and_floor_cloudy_pixels():
    dates = pandas.date_range("2018-04-01", "2018-10-31", periods=40).strftime("%Y-%m-%d")
    t = gee.normalized_time(pandas.Series(dates), crop="maize", year=2018).to_numpy()
    coefficients = numpy.array([3.0, -1.0, 0.5, 0.2, -0.1, 0.05, 0.02])
    gcvi = harmonics.design(t) @ coefficients
    pixels = _pixels(-93.62, 42.03, 2, dates=numpy.array(dates))
    pixels.green[:] = 0.1
    pixels.nir[:] = 0.1 * (gcvi + 1)
    # pixel 1 under cloud on all but 19 dates: under the target floor of 20
    pixels.green[1, 19:] = numpy.nan
    # pixel 2 with a dark date gee.clean drops, still 39 dates
    pixels.green[2, 0] = 0.005
    got = predict.pixel_harmonics(pixels, crop="maize", year=2018)
    assert target.MIN_OBSERVATIONS == 20
    numpy.testing.assert_allclose(got[[0, 2, 3]], numpy.tile(coefficients, (3, 1)), atol=1e-5)
    assert numpy.isnan(got[1]).all()


def test_a_point_takes_its_gridmet_cell():
    cells = weather.Cells(
        values={"pr_5": numpy.array([[1.0, 2.0], [3.0, 4.0]])},
        transform=Affine(0.5, 0, -94.0, 0, -0.5, 43.0),
    )
    got = cells.at(numpy.array([-93.9, -93.1, -93.6]), numpy.array([42.9, 42.1, 42.4]))
    assert got["pr_5"].tolist() == [1.0, 4.0, 3.0]


def test_anchored_keeps_nass_per_county_and_raw_scales_area_by_yield():
    ames, far = _pixels(-93.62, 42.03, 2), _pixels(-95.5, 41.5, 1)
    county = {
        # the unpredicted pixel takes the county mean, 12
        "19169": (ames, numpy.array([10.0, 12.0, numpy.nan, 14.0]), *[numpy.full(4, 11.0)] * 2),
        "19155": (far, numpy.array([9.0]), numpy.array([8.0]), numpy.array([10.0])),
    }
    stats = pandas.DataFrame(
        [["19169", "19-50", 2018, 100.0, 1000.0], ["19155", "19-70", 2018, 50.0, 400.0]],
        columns=["fips", "asd", "year", "area_ha", "production_t"],
    )
    grids = predict.frames(county, stats, {"19169": "19-50", "19155": "19-70"})
    assert all(g.shape == export.FRAME for g in grids.values())
    cell = {f: export.cell_counts(p.crop, p.on)[0] for f, (p, *_) in county.items()}
    assert all(len(c) == 1 for c in cell.values())
    p, h = grids["anchored_P"].ravel(), grids["H"].ravel()
    raw = grids["raw_P"].ravel()
    assert p[cell["19169"]] == 1000.0 and p[cell["19155"]] == 400.0
    assert h.sum() == 150.0
    numpy.testing.assert_allclose(raw[cell["19169"]], 100.0 * 12.0)
    numpy.testing.assert_allclose(raw[cell["19155"]], 50.0 * 9.0)
    numpy.testing.assert_allclose(grids["Ylo"].ravel()[cell["19155"]], 8.0)


def test_features_round_trip_with_their_grid(tmp_path):
    pixels = _pixels(-93.62, 42.03, 3)
    pixels.crop[0, 1] = False
    features = pandas.DataFrame({"a": numpy.arange(8.0), "b": numpy.nan})
    path = tmp_path / "f.npz"
    predict.save_features(features, pixels, path)
    got, crop, grid = predict.load_features(path)
    pandas.testing.assert_frame_equal(got, features.astype("float32"))
    assert (crop == pixels.crop).all()
    assert grid == pixels.on
