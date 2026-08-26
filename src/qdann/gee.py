"""County-level GCVI harmonics from Landsat Collection 2, via Earth Engine.

Ma et al., Remote Sensing of Environment 315 (2024) 114427, Section 2.2. See SPEC.md
section 1. Produces the seven satellite features of every source-domain sample.

The reduction is written once and retrieved two ways, because the interactive path cannot
carry the whole state:

  uv run python -m qdann.gee --crop maize --year 2018 --county 19169   # getInfo, one county
  uv run python -m qdann.gee --crop maize --years 2008-2018 --state 19 --export
  uv run python -m qdann.gee --crop maize --from-csv ~/Downloads/gcvi_maize_19.csv

The middle command writes to the Google Drive of the Earth Engine account, not to this
machine; the third fits Eq. 2 to the CSV once it has been downloaded.
"""

import argparse
import logging
import pathlib

import ee
import pandas

from qdann import env, harmonics

logger = logging.getLogger(__name__)


DATA_DIR = pathlib.Path(__file__).parents[2] / "data"

COUNTIES = "TIGER/2018/Counties"
CDL = "USDA/NASS/CDL"
# single-crop codes only: the double-crop classes (26 winter wheat/soybeans, 225 winter
# wheat/corn, 236 winter wheat/sorghum, 238 winter wheat/cotton) are negligible for Corn
# Belt maize but common for Kansas and Oklahoma wheat. AMBIGUITIES.md #15
CROP_TO_CDL_CODE = {"maize": 1, "soybean": 5, "winter_wheat": 24}

# green and NIR move between sensors, so each is renamed before the collections are merged
SENSOR_BANDS = {
    "LANDSAT/LT05/C02/T1_L2": ("SR_B2", "SR_B4"),
    "LANDSAT/LE07/C02/T1_L2": ("SR_B2", "SR_B4"),
    "LANDSAT/LC08/C02/T1_L2": ("SR_B3", "SR_B5"),
    "LANDSAT/LC09/C02/T1_L2": ("SR_B3", "SR_B5"),
}
# Collection 2 level-2 surface reflectance
SCALE_FACTOR, SCALE_OFFSET = 0.0000275, -0.2
# QA_PIXEL bits 0 fill, 1 dilated cloud, 2 cirrus, 3 cloud, 4 cloud shadow, 5 snow,
# 7 water. Bit 6 is skipped deliberately: it is 1 when the pixel is clear
QA_MASK = 0b10111111
NATIVE_SCALE_M = 30

# a county mean this dark is not cropland, and dividing by it puts GCVI in the hundreds
MIN_GREEN_REFLECTANCE = 0.01


def window(crop: str, year: int) -> tuple[str, str]:
    """The crop's observation window as [start, end), which Eq. 2 normalizes t over.

    Winter wheat is sown in the autumn of the preceding year, so its window straddles two
    calendar years while its CDL year stays the harvest year. SPEC.md section 1.
    """
    if crop == "winter_wheat":
        return f"{year - 1}-09-01", f"{year}-09-01"
    return f"{year}-01-01", f"{year + 1}-01-01"


def reflectance(crop: str, year: int, region: ee.Geometry) -> ee.ImageCollection:
    """Every Tier 1 scene over the window, scaled, cloud-masked and renamed green/nir.

    Tier 2 holds the scenes that failed geometric qualification, whose county footprint is
    misregistered, so only T1 is merged.
    """
    start, end = window(crop=crop, year=year)

    def prepare(image: ee.Image) -> ee.Image:
        scaled = image.select(["green", "nir"]).multiply(SCALE_FACTOR).add(SCALE_OFFSET)
        # surface reflectance is offset-corrected and can fall outside [0, 1] over dark or
        # saturated targets; both bands must survive together or the two means below would
        # be taken over different sets of pixels
        in_range = scaled.gte(0).And(scaled.lte(1)).reduce(ee.Reducer.min())
        clear = image.select("QA_PIXEL").bitwiseAnd(QA_MASK).eq(0)
        return (
            scaled.updateMask(in_range)
            .updateMask(clear)
            .updateMask(image.select("QA_RADSAT").eq(0))
            .set("date", image.date().format("YYYY-MM-dd"))
        )

    collections = [
        ee.ImageCollection(name)
        .filterBounds(region)
        .filterDate(start, end)
        .select(
            [green, nir, "QA_PIXEL", "QA_RADSAT"],
            ["green", "nir", "QA_PIXEL", "QA_RADSAT"],
        )
        .map(prepare)
        for name, (green, nir) in SENSOR_BANDS.items()
    ]
    merged = collections[0]
    for collection in collections[1:]:
        merged = merged.merge(collection)
    return merged


def mosaic_by_date(collection: ee.ImageCollection) -> ee.ImageCollection:
    """Collapse the scenes acquired on one day into a single image.

    A county is not one WRS-2 scene -- Iowa spans paths 25-31 and rows 30-32, so a county
    on a row seam is covered by two images acquired the same day. Reducing per scene would
    give that county two partial means for one date, which enter Eq. 2 as two observations
    at the same t and leave its design matrix rank-deficient.
    """
    joined = ee.Join.saveAll("same_day").apply(
        primary=collection.distinct("date"),
        secondary=collection,
        condition=ee.Filter.equals(leftField="date", rightField="date"),
    )

    def merge(feature: ee.Element) -> ee.Image:
        date = ee.String(feature.get("date"))
        # mosaic() returns an image carrying no properties at all, so the date has to be
        # put back or there is nothing downstream to build t from
        return (
            ee.ImageCollection(ee.List(feature.get("same_day")))
            .mosaic()
            .set("date", date, "system:time_start", ee.Date(date).millis())
        )

    return ee.ImageCollection(joined.map(merge))


def county_means(
    collection: ee.ImageCollection, counties: ee.FeatureCollection, year: int
) -> ee.FeatureCollection:
    """Mean green and NIR per (county, date), as a long table.

    Section 2.2 asks for the mean reflectance of each band and computes GCVI afterwards.
    GCVI is a ratio, so mean(NIR)/mean(green) - 1 is not mean(NIR/green - 1) and the order
    is not interchangeable. AMBIGUITIES.md #14.
    """

    def per_image(element: ee.ComputedObject) -> ee.FeatureCollection:
        image = ee.Image(element)

        def per_county(county: ee.Feature) -> ee.Feature:
            means = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=county.geometry(),
                scale=NATIVE_SCALE_M,
                # the default 1e7 clears an Iowa county's ~1.6M pixels but not a large
                # Texas county's ~16M. bestEffort is never the answer here: it silently
                # coarsens the scale instead, resampling the categorical CDL mask
                maxPixels=int(1e13),
                tileScale=4,
            )
            return ee.Feature(
                None,
                means.combine(
                    {
                        "fips": county.get("GEOID"),
                        "date": image.get("date"),
                        "year": year,
                    }
                ),
            )

        return counties.map(per_county)

    images = collection.toList(collection.size())
    return ee.FeatureCollection(images.map(per_image)).flatten()


def build(crop: str, year: int, counties: ee.FeatureCollection) -> ee.FeatureCollection:
    """The whole server-side reduction: one long table of (fips, year, date, green, nir)."""
    region = counties.geometry()
    cropland = (
        ee.ImageCollection(CDL)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .first()
        .select("cropland")
        .eq(CROP_TO_CDL_CODE[crop])
    )
    dates = mosaic_by_date(reflectance(crop=crop, year=year, region=region))
    # the mask goes on the date mosaic rather than on each scene: once per date instead of
    # once per path/row, and it stays out of the sensor merge. CDL is categorical and sits
    # on a different grid to Landsat, so it must never be resample()d -- nearest neighbour
    # is Earth Engine's default for an integer band and is the only correct choice
    masked = dates.map(
        lambda image: ee.Image(
            image.updateMask(cropland).copyProperties(
                image, ["date", "system:time_start"]
            )
        )
    )
    return county_means(collection=masked, counties=counties, year=year)


def state_counties(
    state_fips: str, geoids: tuple[str, ...] = ()
) -> ee.FeatureCollection:
    ret = ee.FeatureCollection(COUNTIES).filter(ee.Filter.eq("STATEFP", state_fips))
    return ret.filter(ee.Filter.inList("GEOID", list(geoids))) if geoids else ret


def to_frame(features: dict) -> pandas.DataFrame:
    """Turn a getInfo'd or exported feature collection into the long table."""
    return clean(pandas.DataFrame([f["properties"] for f in features["features"]]))


def clean(long: pandas.DataFrame) -> pandas.DataFrame:
    """Drop the rows Eq. 2 cannot use and attach GCVI.

    A county-date whose pixels are entirely masked reduces to a feature with green and nir
    simply absent -- not an error and not a NaN -- so those have to go before the fit sees
    a shorter series than it counts.
    """
    absent = [name for name in ("green", "nir") if name not in long.columns]
    ret = long.reindex(columns=[*long.columns, *absent])
    ret = ret.dropna(subset=["green", "nir"])
    ret = ret[ret["green"] > MIN_GREEN_REFLECTANCE]
    # Eq. 1
    return ret.assign(
        gcvi=ret["nir"] / ret["green"] - 1, fips=ret["fips"].astype(str)
    ).loc[:, ["fips", "year", "date", "green", "nir", "gcvi"]]


def normalized_time(dates: pandas.Series, crop: str, year: int) -> pandas.Series:
    start, end = (pandas.Timestamp(t) for t in window(crop=crop, year=year))
    return (pandas.to_datetime(dates) - start) / (end - start)


def fit_table(long: pandas.DataFrame, crop: str) -> pandas.DataFrame:
    """Fit Eq. 2 per (county, year), returning seven coefficient columns plus a count."""
    names = ["c", "a1", "b1", "a2", "b2", "a3", "b3"]
    rows = []
    for (fips, year), group in long.groupby(["fips", "year"], sort=True):
        # the cheap proof that mosaic_by_date actually ran: two scenes of one county on one
        # day would arrive here as two rows sharing a date
        assert not group["date"].duplicated().any(), (
            f"{fips} {year} has a repeated date"
        )
        coefficients = harmonics.fit(
            t=normalized_time(group["date"], crop=crop, year=int(year)).to_numpy(),
            y=group["gcvi"].to_numpy(),
        )
        rows.append(
            {"fips": fips, "year": int(year), "n_observations": len(group)}
            | dict(zip(names, coefficients, strict=True))
        )
    return pandas.DataFrame(rows, columns=["fips", "year", "n_observations", *names])


def export(crop: str, years: range, state_fips: str) -> ee.batch.Task:
    tables = [
        build(crop=crop, year=year, counties=state_counties(state_fips))
        for year in years
    ]
    merged = tables[0]
    for table in tables[1:]:
        merged = merged.merge(table)
    name = f"gcvi_{crop}_{state_fips}"
    task = ee.batch.Export.table.toDrive(
        collection=merged,
        description=name,
        folder="qdann",
        fileNamePrefix=name,
        fileFormat="CSV",
        selectors=["fips", "year", "date", "green", "nir"],
    )
    task.start()
    return task


def year_range(text: str) -> range:
    first, _, last = text.partition("-")
    return range(int(first), int(last or first) + 1)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop", choices=sorted(CROP_TO_CDL_CODE), required=True)
    parser.add_argument("--year", type=int)
    parser.add_argument("--years", type=year_range)
    parser.add_argument("--state", help="two-digit state FIPS, e.g. 19 for Iowa")
    parser.add_argument("--county", help="five-digit county FIPS, e.g. 19169")
    parser.add_argument("--from-csv", type=pathlib.Path)
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()

    if args.from_csv is not None:
        long = clean(pandas.read_csv(args.from_csv, dtype={"fips": str}))
        table = fit_table(long=long, crop=args.crop)
        out = DATA_DIR / f"gcvi_{args.crop}.parquet"
        out.parent.mkdir(exist_ok=True)
        table.to_parquet(out, index=False)
        print(table.to_string(index=False))
        print(f"\n{len(table):d} county-years -> {out}")
        print(table["n_observations"].describe().to_string())
        return 0

    ee.Initialize(project=env.get("EE_PROJECT"))

    if args.export:
        task = export(crop=args.crop, years=args.years, state_fips=args.state)
        print(
            f"started Drive export {task.id}; watch it at https://code.earthengine.google.com/tasks"
        )
        return 0

    state_fips = args.state or args.county[:2]
    counties = state_counties(state_fips, geoids=(args.county,) if args.county else ())
    long = to_frame(build(crop=args.crop, year=args.year, counties=counties).getInfo())
    print(long.sort_values("date").to_string(index=False))
    print(f"\nGCVI {long['gcvi'].min():.2f} to {long['gcvi'].max():.2f}")
    print(fit_table(long=long, crop=args.crop).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
