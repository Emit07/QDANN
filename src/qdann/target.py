"""The unlabeled target domain: CDL-masked crop pixels, same counties and years as source.

Ma et al., Remote Sensing of Environment 315 (2024) 114427, Section 3.1. Source and target
differ in one thing only -- a county mean versus a single 30 m pixel -- so this leg reuses
`gee.reflectance` and `gee.mosaic_by_date` unchanged and replaces just the final reduction.

  uv run python -m qdann.target --crop maize --year 2018 --county 19169 --per-county 5
  uv run python -m qdann.target --crop maize --years 2008-2018 --state 19 --export

The second writes one CSV per year to the Google Drive of the Earth Engine account, not to
this machine.
"""

import argparse
import logging

import ee
import pandas

from qdann import env, gee

logger = logging.getLogger(__name__)


KEY_COLUMNS = ("fips", "pid", "year")
# the sample size of the Phase-5 county aggregation, not a model hyperparameter
PER_COUNTY = 50


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


def literal(sampled: dict) -> ee.FeatureCollection:
    """The getInfo'd points as plain data, so `stratifiedSample` runs once, not per date."""
    return ee.FeatureCollection(
        [
            ee.Feature(
                ee.Geometry.Point(feature["geometry"]["coordinates"]),
                {key: feature["properties"][key] for key in KEY_COLUMNS},
            )
            for feature in sampled["features"]
        ]
    )


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
        sample_points(
            crop=crop, year=year, counties=counties, per_county=per_county, seed=seed
        ).getInfo()
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


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop", choices=sorted(gee.CROP_TO_CDL_CODE), required=True)
    parser.add_argument("--year", type=int)
    parser.add_argument("--years", type=gee.year_range)
    parser.add_argument("--state", help="two-digit state FIPS, e.g. 19 for Iowa")
    parser.add_argument("--county", help="five-digit county FIPS, e.g. 19169")
    parser.add_argument("--per-county", type=int, default=PER_COUNTY)
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()

    ee.Initialize(project=env.get("EE_PROJECT"))

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

    state_fips = args.state or args.county[:2]
    counties = gee.state_counties(
        state_fips, geoids=(args.county,) if args.county else ()
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
