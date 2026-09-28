"""County-level GCVI harmonics from Landsat Collection 2, via Microsoft Planetary Computer.

The same features as `gee`, read from MPC's free STAC catalogue instead of Earth Engine, so
the pipeline runs without an Earth Engine quota or a Cloud bucket. The masking, mosaic and
fitting rules are `gee`'s, reused rather than restated.
"""

import pathlib
import urllib.request

import pyogrio.raw
import shapely

from qdann import gee

# the vintage Earth Engine's TIGER/2018/Counties was built from, in NAD83 (EPSG:4269)
TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER2018/COUNTY/tl_2018_us_county.zip"
TIGER_PATH = gee.DATA_DIR / "tl_2018_us_county.zip"


def download(url: str, path: pathlib.Path) -> pathlib.Path:
    """Fetch `url` to `path` once; a partial download never takes the final name."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, part)
        part.rename(path)
    return path


def state_counties(
    state_fips: str, geoids: tuple[str, ...] = ()
) -> dict[str, shapely.Geometry]:
    """County polygons by GEOID, in EPSG:4269, mirroring `gee.state_counties`."""
    # STATEFP has to be read for the filter to see it: GDAL ignores the unread fields,
    # and a filter on one then matches nothing rather than raising. The fields come back in
    # file order, not `columns` order, so GEOID is found by name
    meta, _, wkb, fields = pyogrio.raw.read(
        download(TIGER_URL, TIGER_PATH),
        columns=["GEOID", "STATEFP"],
        where=f"STATEFP = '{state_fips}'",
    )
    ids = fields[list(meta["fields"]).index("GEOID")]
    ret = dict(zip(ids, shapely.from_wkb(wkb)))
    return {g: ret[g] for g in geoids} if geoids else ret
