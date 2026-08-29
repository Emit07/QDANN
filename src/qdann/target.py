"""The unlabeled target domain: CDL-masked crop pixels, same counties and years as source.

Ma et al., Remote Sensing of Environment 315 (2024) 114427, Section 3.1. Source and target
differ in one thing only -- a county mean versus a single 30 m pixel -- so this leg reuses
`gee.reflectance` and `gee.mosaic_by_date` unchanged and replaces just the final reduction.

  uv run python -m qdann.target --crop maize --year 2018 --county 19169 --per-county 5
  uv run python -m qdann.target --crop maize --years 2008-2018 --state 19 --export
  uv run python -m qdann.target --crop maize --from-csv ~/Downloads
  uv run python -m qdann.target --crop maize --weather
  uv run python -m qdann.target --crop maize --join

The second writes one CSV per year to the Google Drive of the Earth Engine account, not to
this machine; the third fits Eq. 2 to those CSVs once downloaded, the fourth reads gridMET
at the same points, and the fifth joins the two into the unlabelled target table.
"""

import argparse
import logging
import pathlib

import ee
import pandas

from qdann import env, gee, weather

logger = logging.getLogger(__name__)


KEY_COLUMNS = ("fips", "pid", "year")
# the sample size of the Phase-5 county aggregation, not a model hyperparameter
PER_COUNTY = 50

# Eq. 2 has seven coefficients, so seven distinct dates interpolate with zero residual and
# `harmonics.fit`'s own floor of 7 buys nothing but a fit. The county table says how much
# more is needed: across its worst-observed county-years (15-22 observations) the third
# harmonic's b3 has SD 0.482 against 0.217 in its best-observed 500 -- 2.2x the spread, all
# of it fit noise. Unfiltered, that noise would enter the target's feature variance and the
# adversarial branch would spend itself removing it instead of the aggregation shift.
# AMBIGUITIES.md #18
MIN_OBSERVATIONS = 20
# printed by --from-csv so the floor above is chosen against the retained fraction, not
# against this file
CANDIDATE_FLOORS = (7, 12, 15, 20, 25, 30)


def sample_points(
    crop: str,
    year: int,
    counties: ee.FeatureCollection,
    per_county: int = PER_COUNTY,
    seed: int = 0,
) -> ee.FeatureCollection:
    """`per_county` random crop-pixel centres in each county, drawn fresh for the year.

    `stratifiedSample` rather than `sample`: `sample` draws before masking, so a county
    that is 55% maize returns ~55% of the requested points and the count varies county to
    county -- which silently reweights the county aggregation toward high-maize counties.
    The seed carries the year because CDL rotates: a 2018 maize pixel is 2019 soybean, so a
    panel fixed across years would be wrong.
    """
    cropland = gee.cropland_mask(crop=crop, year=year)

    def per_county_sample(element: ee.ComputedObject) -> ee.FeatureCollection:
        county = ee.Feature(element)
        fips = ee.String(county.get("GEOID"))
        points = cropland.stratifiedSample(
            numPoints=per_county,
            classBand="cropland",
            region=county.geometry(),
            scale=gee.NATIVE_SCALE_M,
            seed=seed + year,
            # zero points of class 0: every sampled pixel is the crop
            classValues=[0, 1],
            classPoints=[0, per_county],
            dropNulls=True,
            tileScale=4,
            geometries=True,
        )
        # sampleRegions, merge and flatten all reassign system:index, so pixel identity has
        # to become an ordinary property here, while it still means something
        return points.map(
            lambda point: point.set(
                "fips",
                fips,
                "year",
                year,
                "pid",
                fips.cat("_")
                .cat(ee.Number(year).format("%d"))
                .cat("_")
                .cat(point.get("system:index")),
            )
        )

    return ee.FeatureCollection(
        counties.toList(counties.size()).map(per_county_sample)
    ).flatten()


def point_values(
    collection: ee.ImageCollection, points: ee.FeatureCollection
) -> ee.FeatureCollection:
    """Green and NIR at each point on each date: the pixel twin of `gee.county_means`.

    No mean, so Eq. 1's ordering question (AMBIGUITIES.md #14) does not arise here; GCVI is
    formed from the pixel's own two bands in `gee.clean`, exactly as for a county.
    """

    def per_image(element: ee.ComputedObject) -> ee.FeatureCollection:
        image = ee.Image(element)
        # sampleRegions drops points whose pixel is masked, which is the cloud screen
        return image.sampleRegions(
            collection=points,
            properties=list(KEY_COLUMNS),
            scale=gee.NATIVE_SCALE_M,
            tileScale=4,
        ).map(lambda feature: feature.set("date", image.get("date")))

    images = collection.toList(collection.size())
    return ee.FeatureCollection(images.map(per_image)).flatten()


def literal(features: list[dict]) -> ee.FeatureCollection:
    """The getInfo'd points as plain data, so `stratifiedSample` runs once, not per date."""
    return ee.FeatureCollection(
        [
            ee.Feature(
                ee.Geometry.Point(feature["geometry"]["coordinates"]),
                {key: feature["properties"][key] for key in KEY_COLUMNS},
            )
            for feature in features
        ]
    )


def fetch_points(
    crop: str,
    year: int,
    counties: ee.FeatureCollection,
    per_county: int = PER_COUNTY,
    seed: int = 0,
    chunk: int = 8,
) -> list[dict]:
    """The sampled points as plain GeoJSON, a few counties per request.

    `stratifiedSample` is one aggregation per county and the interactive endpoint caps how
    many may run at once: all 99 Iowa counties in one request answers `Too many concurrent
    aggregations`, eight of them return in a second. Both client-side legs draw their points
    through here, with the same seed, and `join` asserts the two agree.
    """
    geoids = counties.aggregate_array("GEOID").getInfo()
    ret = []
    for start in range(0, len(geoids), chunk):
        part = counties.filter(
            ee.Filter.inList("GEOID", geoids[start : start + chunk])
        )
        sampled = sample_points(
            crop=crop, year=year, counties=part, per_county=per_county, seed=seed
        ).getInfo()
        ret.extend(sampled["features"])
    return ret


def fetch(
    crop: str,
    year: int,
    counties: ee.FeatureCollection,
    per_county: int = PER_COUNTY,
    seed: int = 0,
    chunk: int = 8,
) -> pandas.DataFrame:
    """The interactive path, a few dates per request.

    One date's `sampleRegions` is one aggregation and the interactive endpoint caps how
    many may run concurrently -- a whole year is ~66 dates and returns `Too many concurrent
    aggregations`, as does sampling all 99 counties at once. The batch export has no such
    cap and computes the year in one go, so this chunking is the interactive path's problem
    alone.
    """
    points = literal(
        fetch_points(
            crop=crop, year=year, counties=counties, per_county=per_county, seed=seed
        )
    )
    dates = gee.mosaic_by_date(
        gee.reflectance(crop=crop, year=year, region=counties.geometry())
    )
    images = dates.toList(dates.size())
    n_dates = int(dates.size().getInfo())
    frames = []
    for start in range(0, n_dates, chunk):
        part = ee.ImageCollection(images.slice(start, start + chunk))
        features = point_values(collection=part, points=points).getInfo()
        frames.append(pandas.DataFrame([f["properties"] for f in features["features"]]))
        logger.info(f"dates {start:d}-{min(start + chunk, n_dates):d}: {len(frames[-1]):d} rows")
    return pandas.concat(frames, ignore_index=True)


def build(
    crop: str,
    year: int,
    counties: ee.FeatureCollection,
    per_county: int = PER_COUNTY,
    seed: int = 0,
) -> ee.FeatureCollection:
    """One long table of (fips, pid, year, date, green, nir) over the sampled pixels.

    The imagery is deliberately *not* `updateMask(cropland)`ed the way `gee.build` masks
    it: the sampling already is the mask. `stratifiedSample` takes no projection argument,
    so it returns CDL pixel centres on CDL's Albers grid and `sampleRegions` reads whichever
    Landsat pixel (UTM, offset by up to half a pixel) contains each. Re-applying the mask
    would make Earth Engine resample CDL onto the Landsat grid and drop every point whose
    Landsat centre falls in the neighbouring CDL cell -- a non-random thinning at exactly
    the field edges. Only crop locations contribute to either domain either way: the source
    masks then averages, the target selects then reads.
    """
    region = counties.geometry()
    dates = gee.mosaic_by_date(gee.reflectance(crop=crop, year=year, region=region))
    points = sample_points(
        crop=crop, year=year, counties=counties, per_county=per_county, seed=seed
    )
    return point_values(collection=dates, points=points)


def export(
    crop: str, year: int, state_fips: str, per_county: int = PER_COUNTY
) -> ee.batch.Task:
    """One task per year: ~99 counties x 50 points x ~40 dates is already ~200k rows.

    `selectors` is not optional. Without it Earth Engine infers the columns from the first
    feature and writes a `.geo` column -- 200k point geometries per year, for nothing.
    """
    name = f"target_{crop}_{state_fips}_{year:d}"
    task = ee.batch.Export.table.toDrive(
        collection=build(
            crop=crop,
            year=year,
            counties=gee.state_counties(state_fips),
            per_county=per_county,
        ),
        description=name,
        folder="qdann",
        fileNamePrefix=name,
        fileFormat="CSV",
        selectors=["fips", "pid", "year", "date", "green", "nir"],
    )
    task.start()
    return task


def weather_values(
    crop: str, year: int, points: ee.FeatureCollection
) -> ee.FeatureCollection:
    """The twenty gridMET features at each point: `weather.county_means` without the mean.

    SPEC.md section 1 has the paper resampling gridMET to 30 m for the subfield leg. Read at
    gridMET's own scale a point simply takes its containing cell's value, which is the same
    number the upsample would have produced, without materializing a 4 km grid at 30 m.
    """
    return weather.monthly_means(crop=crop, year=year).sampleRegions(
        collection=points,
        properties=list(KEY_COLUMNS),
        scale=weather.native_scale(),
        tileScale=4,
    )


def fetch_weather(
    crop: str,
    years: range,
    counties: ee.FeatureCollection,
    per_county: int = PER_COUNTY,
    seed: int = 0,
) -> pandas.DataFrame:
    """One row per pixel-year: twenty weather columns plus the point's own lon/lat.

    lon/lat are carried as ordinary columns rather than as an exported `.geo`: they exist
    for a future map, and a GeoJSON column inside a CSV is a parsing problem for later.
    """
    frames = []
    for year in years:
        points = fetch_points(
            crop=crop, year=year, counties=counties, per_county=per_county, seed=seed
        )
        sampled = weather_values(
            crop=crop, year=year, points=literal(points)
        ).getInfo()
        frame = pandas.DataFrame([f["properties"] for f in sampled["features"]])
        frames.append(
            frame.merge(
                pandas.DataFrame(
                    [
                        {
                            "pid": point["properties"]["pid"],
                            "lon": point["geometry"]["coordinates"][0],
                            "lat": point["geometry"]["coordinates"][1],
                        }
                        for point in points
                    ]
                ),
                on="pid",
                how="left",
            )
        )
        logger.info(f"{year:d}: {len(frames[-1]):d} of {len(points):d} points")
    ret = pandas.concat(frames, ignore_index=True)
    ret["fips"] = ret["fips"].astype(str)
    return ret.loc[
        :, [*KEY_COLUMNS, *weather.weather_columns(crop), "lon", "lat"]
    ].sort_values(list(KEY_COLUMNS))


def read_csvs(path: pathlib.Path, crop: str) -> pandas.DataFrame:
    """The exported CSVs, given a directory or a glob rather than one file.

    Eleven per-year tasks write eleven files, and Earth Engine shards a large table into
    numbered parts on top of that. `fips` and `pid` are strings: 19001 read as an integer
    loses the leading zero of every state below 10 and never joins again.
    """
    paths = sorted(
        path.glob(f"target_{crop}_*.csv")
        if path.is_dir()
        else path.parent.glob(path.name)
    )
    if not paths:
        raise SystemExit(f"no exported CSVs under {path}")
    logger.info(f"reading {len(paths):d} files: {', '.join(p.name for p in paths)}")
    return pandas.concat(
        [pandas.read_csv(p, dtype={"fips": str, "pid": str}) for p in paths],
        ignore_index=True,
    )


def retained(long: pandas.DataFrame) -> pandas.DataFrame:
    """The fraction of pixel-years each candidate floor would keep. AMBIGUITIES.md #18."""
    counts = long.groupby(list(KEY_COLUMNS)).size()
    return pandas.DataFrame(
        {
            "floor": CANDIDATE_FLOORS,
            "pixel_years": [int((counts >= f).sum()) for f in CANDIDATE_FLOORS],
            "retained": [float((counts >= f).mean()) for f in CANDIDATE_FLOORS],
        }
    )


def join(
    gcvi: pandas.DataFrame, weather_table: pandas.DataFrame, crop: str
) -> pandas.DataFrame:
    """One row per pixel-year: the same 27 features as the source, and no label.

    Both legs re-derive `sample_points` with the same seed, so every fitted pixel must find
    its weather row -- unlike the source join there is no Title 7 suppression to tolerate
    here, and a missing row would mean the sampling was not deterministic. The reverse
    direction is expected to lose rows: a pixel under cloud on too many dates fails the
    observation floor and has no harmonics at all.
    """
    keys = list(KEY_COLUMNS)
    fitted = gcvi.dropna(subset=list(weather.HARMONIC_COLUMNS))
    ret = fitted.loc[:, [*keys, *weather.HARMONIC_COLUMNS]].merge(
        weather_table, on=keys, how="inner"
    )
    orphans = set(map(tuple, fitted[keys].to_numpy())) - set(
        map(tuple, weather_table[keys].to_numpy())
    )
    assert not orphans, (
        f"{len(orphans):d} fitted pixel-years have no weather row -- the two legs sampled "
        f"different points: {sorted(orphans)[:5]}"
    )
    logger.info(
        f"{len(gcvi) - len(fitted):d} of {len(gcvi):d} pixel-years dropped by the "
        f"observation floor"
    )
    return ret.loc[:, [*keys, *weather.HARMONIC_COLUMNS, *weather.weather_columns(crop)]]


def compare(
    target: pandas.DataFrame, source: pandas.DataFrame, crop: str
) -> pandas.DataFrame:
    """Each feature's first two moments in both domains, the cheapest bug-catcher there is.

    A county mean is an average of its own pixels, so the centres should nearly coincide and
    the pixel SDs should be the larger -- that widening *is* the aggregation shift the
    adversarial branch exists to absorb. Read the failures off the same two columns:
    `mean_shift_sd` beyond an SD or two points at the mask or the window rather than at
    aggregation, a large `sd_ratio` points back at the observation floor, and any weather
    row that is not near 0 and 1 is a pipeline bug -- a pixel and its county read the same
    4 km gridMET cells.
    """
    columns = [*weather.HARMONIC_COLUMNS, *weather.weather_columns(crop)]
    return pandas.DataFrame(
        {
            "source_mean": source[columns].mean(),
            "target_mean": target[columns].mean(),
            "source_sd": source[columns].std(),
            "target_sd": target[columns].std(),
        }
    ).assign(
        mean_shift_sd=lambda frame: (frame["target_mean"] - frame["source_mean"])
        / frame["source_sd"],
        sd_ratio=lambda frame: frame["target_sd"] / frame["source_sd"],
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop", choices=sorted(gee.CROP_TO_CDL_CODE), required=True)
    parser.add_argument("--year", type=int)
    parser.add_argument(
        "--years", type=gee.year_range, default=gee.year_range("2008-2018")
    )
    parser.add_argument(
        "--state", default="19", help="two-digit state FIPS, e.g. 19 for Iowa"
    )
    parser.add_argument("--county", help="five-digit county FIPS, e.g. 19169")
    parser.add_argument("--per-county", type=int, default=PER_COUNTY)
    parser.add_argument("--export", action="store_true")
    parser.add_argument(
        "--from-csv", type=pathlib.Path, help="a directory or glob of exported CSVs"
    )
    parser.add_argument(
        "--min-observations", type=int, default=MIN_OBSERVATIONS, help="Eq. 2's floor"
    )
    parser.add_argument(
        "--weather", action="store_true", help="read gridMET at the sampled points"
    )
    parser.add_argument(
        "--join", action="store_true", help="skip Earth Engine and build the target table"
    )
    args = parser.parse_args()

    gcvi_out = weather.DATA_DIR / f"gcvi_target_{args.crop}.parquet"
    gridmet_out = weather.DATA_DIR / f"gridmet_target_{args.crop}.parquet"
    weather.DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.from_csv is not None:
        long = gee.clean(read_csvs(args.from_csv, crop=args.crop), keys=KEY_COLUMNS)
        print(retained(long).to_string(index=False))
        table = gee.fit_table(
            long=long,
            crop=args.crop,
            keys=KEY_COLUMNS,
            min_observations=args.min_observations,
        )
        table.to_parquet(gcvi_out, index=False)
        print(table["n_observations"].describe().to_string())
        fitted = table.dropna(subset=list(weather.HARMONIC_COLUMNS))
        print(
            f"\n{len(table):d} pixel-years, {len(fitted):d} above the floor of "
            f"{args.min_observations:d} -> {gcvi_out}"
        )
        return 0

    if args.join:
        table = join(
            gcvi=weather.read(gcvi_out),
            weather_table=weather.read(gridmet_out),
            crop=args.crop,
        )
        out = weather.DATA_DIR / f"target_{args.crop}.parquet"
        table.to_parquet(out, index=False)
        print(
            compare(
                target=table,
                source=weather.read(weather.DATA_DIR / f"source_{args.crop}.parquet"),
                crop=args.crop,
            ).to_string()
        )
        print("\npixels per county-year")
        print(table.groupby(["fips", "year"]).size().describe().to_string())
        counties = table.groupby("year")["fips"].nunique()
        print(f"\ncounties per year {counties.min():d} to {counties.max():d}")
        print(
            f"\n{len(table):d} pixel-years, "
            f"{len(table.columns) - len(KEY_COLUMNS):d} features -> {out}"
        )
        return 0

    ee.Initialize(project=env.get("EE_PROJECT"))

    if args.weather:
        table = fetch_weather(
            crop=args.crop,
            years=args.years,
            counties=gee.state_counties(args.state),
            per_county=args.per_county,
        )
        table.to_parquet(gridmet_out, index=False)
        print(table.describe().to_string())
        print(f"\n{len(table):d} pixel-years -> {gridmet_out}")
        return 0

    if args.export:
        for year in args.years:
            task = export(
                crop=args.crop,
                year=year,
                state_fips=args.state,
                per_county=args.per_county,
            )
            print(f"{year:d}: started Drive export {task.id}")
        print("watch them at https://code.earthengine.google.com/tasks")
        return 0

    counties = gee.state_counties(
        args.county[:2] if args.county else args.state,
        geoids=(args.county,) if args.county else (),
    )
    # a wrong classBand is worth five seconds here rather than an hour into a batch task
    bands = gee.cropland_mask(crop=args.crop, year=args.year).bandNames().getInfo()
    print(f"mask bands {bands}")
    long = gee.clean(
        fetch(
            crop=args.crop,
            year=args.year,
            counties=counties,
            per_county=args.per_county,
        ),
        keys=KEY_COLUMNS,
    )
    print(long.sort_values(["pid", "date"]).to_string(index=False))
    # the same-day mosaic's proof on this path, and the number Phase 3's floor turns on
    assert not long.duplicated(["pid", "date"]).any(), "a pixel has a repeated date"
    print(f"\nGCVI {long['gcvi'].min():.2f} to {long['gcvi'].max():.2f}")
    print("\nobservations per pixel-year")
    print(long.groupby("pid").size().describe().to_string())
    print(gee.fit_table(long=long, crop=args.crop, keys=KEY_COLUMNS).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
