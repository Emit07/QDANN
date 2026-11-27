"""Checks on the Planetary Computer leg that run without the network.

The reads themselves are checked against Earth Engine county-years (AMBIGUITIES.md,
the MPC plan). What is pinned here is everything that would corrupt the means silently:
which pixels count as clear, which scene wins a same-day overlap, the grid the county is
read on, which pixels are the county's, and what a warp does across UTM zones.
"""

import numpy
import pandas
import pyproj
import pytest
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
    assert (row["one_sided"], row["one_sided_clear"]) == (0, 0)
    assert row["rel_median"] == row["rel_p95"] == 0
    assert row["coef_all"] > mpc.MAX_COEF_SD > row["coef_filtered"]
    assert mpc.gate(pandas.DataFrame([row, *clean])) == []

    # the same miss on a date clear enough for `fit` to keep is a real disagreement
    mpc_raw.loc[len(mpc_raw) - 1, "n_clear"] = 600
    row = mpc.compare(gee_raw, mpc_raw, sd, "maize")
    assert (row["one_sided"], row["one_sided_clear"]) == (1, 600)
    assert mpc.gate(pandas.DataFrame([row, *clean])) == [
        "B: a one-sided date has >= 100 clear pixels"
    ]


def test_a_target_point_reads_the_landsat_pixel_containing_it():
    values = numpy.arange(40 * 40, dtype=numpy.uint16).reshape(40, 40)
    crs = rasterio.CRS.from_epsg(32615)
    transform = Affine(30, 0, 400005, 0, -30, 4650015)
    rows, cols = numpy.array([0, 7, 7, 39]), numpy.array([0, 12, 13, 39])
    # 14 m off each centre toward a corner, where a resampled read would pick a neighbour
    x, y = 400005 + 30 * (cols + 0.5), 4650015 - 30 * (rows + 0.5)
    x, y = [*(x + [14, -14, 14, -14]), 390000], [*(y + [14, 14, -14, -14]), 4650000]
    lon, lat = rasterio.warp.transform(crs, "EPSG:4326", x, y)
    with _raster(values, crs, transform) as src:
        read = mpc.sample(src, numpy.array(lon), numpy.array(lat))
    assert read.tolist() == [*values[rows, cols].tolist(), 65535]


def test_read_years_collects_county_dirs_and_rejects_repeats(tmp_path):
    row = {"fips": "19001", "year": 2018, "date": "2018-06-01", "green": 0.1, "nir": 0.3}
    for fips in ("19001", "19003"):
        (tmp_path / fips).mkdir()
        pandas.DataFrame([row | {"fips": fips}]).to_parquet(
            tmp_path / fips / "gcvi_mpc_maize_2018.parquet"
        )
    long = mpc.read_years(tmp_path, prefix="gcvi_mpc", crop="maize", keys=gee.KEY_COLUMNS)
    assert sorted(long["fips"]) == ["19001", "19003"]

    # the same county read again in a second directory
    (tmp_path / "again").mkdir()
    pandas.DataFrame([row]).to_parquet(tmp_path / "again" / "gcvi_mpc_maize_2018.parquet")
    with pytest.raises(AssertionError, match="repeats"):
        mpc.read_years(tmp_path, prefix="gcvi_mpc", crop="maize", keys=gee.KEY_COLUMNS)


def test_fit_drops_the_sliver_dates(tmp_path):
    dates = pandas.date_range("2018-03-01", "2018-11-01", freq="8D").strftime(
        "%Y-%m-%d"
    )
    t = numpy.linspace(0, 1, len(dates))
    long = pandas.DataFrame(
        {
            "fips": "19169",
            "year": 2018,
            "date": dates,
            "green": 0.08,
            "nir": 0.3 + 0.2 * numpy.sin(2 * numpy.pi * t),
            "n_clear": 5000,
            "n_crop": 10000,
        }
    )
    (tmp_path / "19169").mkdir()
    long.to_parquet(tmp_path / "19169" / "gcvi_mpc_maize_2018.parquet")
    expected = mpc.fit(tmp_path, crop="maize")

    # a 40-pixel mean far off the curve, which #20 says is not the county's
    sliver = long.iloc[[3]].assign(date="2018-03-27", nir=0.9, n_clear=40)
    pandas.concat([long, sliver]).to_parquet(
        tmp_path / "19169" / "gcvi_mpc_maize_2018.parquet"
    )
    pandas.testing.assert_frame_equal(mpc.fit(tmp_path, crop="maize"), expected)


def test_years_after_mpc_read_one_cropscape_file_per_state(monkeypatch):
    def fail():
        raise AssertionError("MPC has no CDL after 2021")

    monkeypatch.setattr(mpc, "catalog", fail)
    monkeypatch.setattr(mpc, "cropscape", lambda year, state: f"CDL_{year}_{state}.tif")
    a, b, c = shapely.box(-94, 42, -93, 43), shapely.box(-93, 42, -92, 43), shapely.box(-94, 40, -93, 41)
    tiles = mpc.cdl_tiles(year=2022, counties={"19001": a, "29001": c, "19003": b})
    assert [href for href, _ in tiles] == ["CDL_2022_19.tif", "CDL_2022_29.tif"]
    assert tiles[0][1].equals(shapely.union_all([a, b])) and tiles[1][1].equals(c)


def test_a_state_files_background_beyond_its_border_keeps_the_neighbours_crop(tmp_path):
    # two state files over one county, each 0 (CDL background) where the other state is:
    # the second must not erase the first's maize
    polygon = shapely.box(-93.5, 42.0, -93.4, 42.05)
    on = mpc.grid(polygon)
    half = on.width // 2
    tiles = []
    for name, cols in (("a", slice(0, half)), ("b", slice(half, None))):
        values = numpy.zeros((on.height, on.width), dtype=numpy.uint8)
        values[:, cols] = gee.CROP_TO_CDL_CODE["maize"]
        path = tmp_path / f"{name}.tif"
        with rasterio.open(
            path, "w", driver="GTiff", height=on.height, width=on.width, count=1,
            dtype="uint8", crs=on.crs, transform=on.transform,
        ) as dst:
            dst.write(values, 1)
        tiles.append((str(path), polygon))
    crop = mpc.cropland("maize", polygon, on, tiles)
    assert crop[mpc.inside(polygon, on)].all()


def test_a_truncated_scene_is_dropped_and_its_date_kept(monkeypatch):
    # MPC serves the odd truncated asset; it fails every retry, and must cost one scene,
    # not the county-year
    import datetime
    import types

    day = datetime.datetime(2022, 7, 23, tzinfo=datetime.timezone.utc)
    good, bad = (types.SimpleNamespace(id=i, datetime=day) for i in ("good", "bad"))
    polygon = shapely.box(-93.5, 42.0, -93.45, 42.03)
    monkeypatch.setattr(mpc.time, "sleep", lambda _: None)
    monkeypatch.setattr(mpc, "scenes", lambda **_: [bad, good])
    monkeypatch.setattr(mpc, "cdl_tiles", lambda **_: [])
    monkeypatch.setattr(mpc, "cropland", lambda **kw: mpc.inside(polygon, kw["on"]))
    monkeypatch.setattr(mpc, "readable", lambda item: item is not bad)

    def date_means(group, *_):
        if bad in group:
            raise rasterio.errors.RasterioIOError("not recognized as being in a supported file format")
        row = {"fips": "19169", "date": "2022-07-23", "green": 0.1, "nir": 0.4, "n_clear": 5, "n_crop": 5}
        return [row], pandas.DataFrame(columns=["fips", "pid", "date", "green", "nir"]), {}

    monkeypatch.setattr(mpc, "date_means", date_means)
    long, ids, _, _ = mpc.county_means("maize", 2022, {"19169": polygon})
    assert ids == ["good"] and list(long["date"]) == ["2022-07-23"]


def test_landsat7_is_left_out_only_after_its_last_year(monkeypatch):
    import datetime
    import types

    def item(platform):
        return types.SimpleNamespace(
            id=platform, datetime=datetime.datetime(2023, 6, 1, tzinfo=datetime.timezone.utc),
            properties={"landsat:collection_category": "T1", "platform": platform},
            geometry=shapely.geometry.mapping(shapely.box(-94, 41, -93, 43)),
        )

    found = types.SimpleNamespace(items=lambda: [item("landsat-7"), item("landsat-8")])
    monkeypatch.setattr(mpc, "catalog", lambda: types.SimpleNamespace(search=lambda **_: found))
    region = shapely.box(-93.6, 42.0, -93.4, 42.1)
    ids = lambda **kw: [i.id for i in mpc.scenes("maize", 2023, region, **kw)]
    assert ids() == ["landsat-7", "landsat-8"]
    assert ids(landsat7_last=2023) == ["landsat-7", "landsat-8"]
    assert ids(landsat7_last=2021) == ["landsat-8"]


def test_write_atomic_leaves_no_part(tmp_path):
    path = tmp_path / "gcvi_mpc_maize_2020.parquet"
    mpc.write_atomic(pandas.DataFrame({"a": [1, 2]}), path)
    assert pandas.read_parquet(path)["a"].tolist() == [1, 2]
    assert [p.name for p in tmp_path.iterdir()] == [path.name]
