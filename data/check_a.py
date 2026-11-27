"""Check A: stored target reflectance is the containing pixel's DN, scaled.

Re-reads each point with rasterio's own `sample` (not mpc.sample) on single-scene dates,
where no mosaic order can pick another scene.
"""

import itertools
import sys

import numpy
import pandas
import rasterio
import rasterio.warp

from qdann import gee, mpc

fips, year, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
stored = pandas.read_parquet(f"{out}/target_mpc_maize_{year}.parquet")
points = mpc.points("maize", year, [fips]).set_index("pid")
items = mpc.scenes("maize", year, mpc.state_counties(fips[:2], (fips,))[fips])
single = [list(g)[0] for _, g in itertools.groupby(items, key=mpc.date)]
single = [i for i in single if sum(mpc.date(j) == mpc.date(i) for j in items) == 1]
single = [i for i in single if (stored["date"] == mpc.date(i)).sum() >= 5][:3]
assert len(single) == 3, "fewer than 3 single-scene dates with 5 clear points"
checked = 0
for item in single:
    day = stored[stored["date"] == mpc.date(item)].head(5)
    with rasterio.Env(**mpc.GDAL_ENV), mpc.opened(item.assets["green"].href) as src:
        xs, ys = rasterio.warp.transform(
            "EPSG:4326", src.crs, points.loc[day["pid"], "lon"].to_numpy(), points.loc[day["pid"], "lat"].to_numpy()
        )
        dn = numpy.array([v[0] for v in src.sample(zip(xs, ys))], dtype=numpy.float64)
    expected = dn * gee.SCALE_FACTOR + gee.SCALE_OFFSET
    assert numpy.allclose(day["green"], expected, rtol=0, atol=1e-6), (item.id, day["green"].tolist(), expected)
    checked += len(day)
    print(item.id, "ok", dn.astype(int).tolist())
print(f"OK: {checked:d} point-dates match the containing pixel")
