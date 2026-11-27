"""One-off: the Tier B GEE per-date references, raw getInfo properties as CSV."""
import pathlib, ee, pandas
from qdann import env, gee
ee.Initialize(project=env.get("EE_PROJECT"))
out = gee.DATA_DIR / "reference_gee" / "tier_b"
for line in (gee.DATA_DIR / "tier_b_subset.txt").read_text().split("\n"):
    if not line:
        continue
    fips, year = line.split()
    path = out / f"gee_maize_{fips}_{year}.csv"
    if path.exists():
        continue
    for attempt in range(5):
        try:
            fc = gee.build(crop="maize", year=int(year), counties=gee.state_counties("19", (fips,))).getInfo()
            break
        except Exception as e:
            print(fips, year, "retry", attempt, e, flush=True)
    frame = pandas.DataFrame([f["properties"] for f in fc["features"]])
    frame[sorted(frame.columns)].to_csv(path, index=False)
    print(fips, year, len(frame), flush=True)
