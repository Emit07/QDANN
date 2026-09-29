"""Checks on the Planetary Computer leg that run without the network.

The reads themselves are checked against Earth Engine county-years (AMBIGUITIES.md,
the MPC plan). What is pinned here is everything that would corrupt the means silently:
which pixels count as clear, which scene wins a same-day overlap, the grid the county is
read on, which pixels are the county's, and what a warp does across UTM zones.
"""

import numpy
import pandas
import pyproj
import rasterio
import rasterio.warp
import shapely
import shapely.ops
from rasterio.io import MemoryFile
from rasterio.transform import Affine

from qdann import gee, mpc

# a DN whose scaled reflectance is 0.075, comfortably inside [0, 1]
DN = round((0.075 - gee.SCALE_OFFSET) / gee.SCALE_FACTOR)


def _reflectance(qa=0, radsat=0, green=DN, nir=DN):
    return mpc.reflectance(
        numpy.array([green], dtype=numpy.uint16),
        numpy.array([nir], dtype=numpy.uint16),
        numpy.array([qa], dtype=numpy.uint16),
        numpy.array([radsat], dtype=numpy.uint16),
    )


def test_a_clear_pixel_is_scaled_to_reflectance():
    green, nir = _reflectance()
    numpy.testing.assert_allclose([green[0], nir[0]], [0.075, 0.075], atol=1e-4)


def test_bit_6_alone_does_not_mask_but_every_other_qa_bit_does():
    # bit 6 is 1 on a clear pixel, so masking it would drop exactly the pixels wanted
    assert not numpy.isnan(_reflectance(qa=1 << 6)[0][0])
    for bit in (0, 1, 2, 3, 4, 5, 7):
        assert numpy.isnan(_reflectance(qa=(1 << 6) | (1 << bit))[0][0]), bit


def test_saturation_and_out_of_range_reflectance_mask_both_bands():
    assert numpy.isnan(_reflectance(radsat=1)).all()
    # DN 0 is fill and scales to -0.2; 65535 scales past 1
    assert numpy.isnan(_reflectance(green=0)).all()
    assert numpy.isnan(_reflectance(nir=65535)).all()


def test_the_last_valid_scene_wins_a_same_day_overlap():
    nan = numpy.nan
    first = (numpy.array([1.0, 1.0, 1.0]), numpy.array([1.0, 1.0, 1.0]))
    later = (numpy.array([2.0, nan, 2.0]), numpy.array([2.0, nan, 2.0]))
    last = (numpy.array([nan, nan, 3.0]), numpy.array([nan, nan, 3.0]))
    green, nir = mpc.mosaic([first, later, last])
    numpy.testing.assert_array_equal(green, [2.0, 1.0, 3.0])
    numpy.testing.assert_array_equal(nir, [2.0, 1.0, 3.0])


def test_a_county_grid_is_landsat_aligned_in_its_centroids_zone():
    story = shapely.box(-93.70, 41.86, -93.23, 42.21)
    woodbury = shapely.box(-96.50, 42.21, -95.86, 42.56)
    assert mpc.utm(story).to_epsg() == 32615
    assert mpc.utm(woodbury).to_epsg() == 32614
    on = mpc.grid(story)
    assert on.transform.c % 30 == 15 and on.transform.f % 30 == 15
    minx, miny, maxx, maxy = mpc.project(story, on.crs).bounds
    left, top = on.transform.c, on.transform.f
    assert left <= minx and top >= maxy
    assert left + 30 * on.width >= maxx and top - 30 * on.height <= miny


def test_a_pixel_is_the_countys_when_its_centre_is():
    on = mpc.Grid(
        crs=rasterio.CRS.from_epsg(32615),
        transform=Affine(30, 0, 0, 0, -30, 60),
        width=2,
        height=2,
    )
    to_4269 = pyproj.Transformer.from_crs(on.crs, mpc.TIGER_CRS, always_xy=True)
    # covers the left column's centres (x = 15) but not the right's (x = 45)
    polygon = shapely.ops.transform(to_4269.transform, shapely.box(0, 0, 40, 60))
    numpy.testing.assert_array_equal(
        mpc.inside(polygon, on), [[True, False], [True, False]]
    )


def _raster(values, crs, transform, nodata=None):
    memfile = MemoryFile()
    with memfile.open(
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype=values.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(values, 1)
    return memfile.open()


def test_a_cross_zone_warp_keeps_categorical_values_and_fills_outside_the_scene():
    # a scene on the zone 16 grid, west of -90 and so read onto a zone 15 grid
    values = numpy.random.default_rng(0).choice([1, 5, 24], size=(40, 40))
    values = values.astype(numpy.uint16)
    scene_crs = rasterio.CRS.from_epsg(32616)
    x, y = rasterio.warp.transform("EPSG:4326", scene_crs, [-90.03], [42.0])
    with _raster(values, scene_crs, Affine(30, 0, x[0], 0, -30, y[0])) as src:
        bounds = rasterio.warp.transform_bounds(scene_crs, "EPSG:4326", *src.bounds)
        polygon = shapely.box(*bounds).buffer(0.01)
        on = mpc.grid(polygon)
        assert on.crs.to_epsg() == 32615
        read = mpc.read(src, on)
    # nearest never invents a class, and the margin outside the scene reads as fill
    assert set(numpy.unique(read)) <= {1, 5, 24, 65535}
    assert (read[0] == 65535).all() and (read[:, 0] == 65535).all()
    assert {1, 5, 24} <= set(numpy.unique(read))


def test_a_valid_zero_survives_the_warp_where_a_zero_fill_would_rewrite_it():
    # CDL background and an unsaturated qa_radsat are both a real 0 with no nodata set
    values = numpy.tile(numpy.array([[0, 5]], dtype=numpy.uint8), (40, 20))
    crs = rasterio.CRS.from_epsg(32615)
    with _raster(values, crs, Affine(30, 0, 400005, 0, -30, 4650015)) as src:
        bounds = rasterio.warp.transform_bounds(crs, "EPSG:4326", *src.bounds)
        read = mpc.read(src, mpc.grid(shapely.box(*bounds).buffer(0.005)))
    counts = dict(zip(*numpy.unique(read, return_counts=True)))
    assert counts[0] == counts[5] == 800 and 1 not in counts


def test_the_gate_forgives_sliver_dates_and_nothing_else():
    dates = pandas.date_range("2018-03-01", "2018-11-01", freq="8D").strftime(
        "%Y-%m-%d"
    )
    t = numpy.linspace(0, 1, len(dates))
    nir = 0.3 + 0.2 * numpy.sin(2 * numpy.pi * t)
    gee_raw = pandas.DataFrame(
        {"date": dates, "fips": "19169", "green": 0.08, "nir": nir, "year": 2018}
    )
    mpc_raw = gee_raw.assign(n_clear=5000, n_crop=10000)
    # a sliver the two grids read differently, and a date only MPC kept, from 40 pixels
    mpc_raw.loc[3, ["nir", "n_clear"]] = [0.9, 100]
    mpc_raw.loc[len(mpc_raw)] = ["2018-11-05", "19169", 0.08, 0.3, 2018, 40, 10000]
    sd = pandas.Series(0.3, index=mpc.COEFFICIENTS)
    # 19 clean county-years, so one mismatch leaves the 95% with identical dates
    clean = [
        mpc.compare(gee_raw, gee_raw.assign(n_clear=5000, n_crop=10000), sd, "maize")
    ] * 19

    row = mpc.compare(gee_raw, mpc_raw, sd, "maize")
    assert (row["one_sided"], row["one_sided_clear"]) == (1, 40)
    assert row["rel_median"] == row["rel_p95"] == 0
    assert row["coef_all"] > mpc.MAX_COEF_SD > row["coef_filtered"]
    assert mpc.gate(pandas.DataFrame([row, *clean])) == []

    # the same miss on a date with 100 clear pixels is a real disagreement
    mpc_raw.loc[len(mpc_raw) - 1, "n_clear"] = 100
    row = mpc.compare(gee_raw, mpc_raw, sd, "maize")
    assert mpc.gate(pandas.DataFrame([row, *clean])) == [
        "B: a one-sided date has >= 100 clear pixels"
    ]
