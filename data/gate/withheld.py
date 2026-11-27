"""Withheld counties against NASS's pooled rows, 2019-2023 (plan Step 8, reviewer's test).

A withheld county-year has no label, so no model ever trained on it. Its county-mean
features (the same MPC reads that built the 2008-2023 table, #27) go through the single-seed
2008-2023 head. Each OTHER (COMBINED) COUNTIES pool (per district in 2019, statewide 19-99
from 2020, #31) is then compared with the mean prediction over its member counties, weighted
by each county's mean published harvested area 2008-2023 (its own area is withheld). The
fallbacks are the ones jdLUC uses: the district yield (2019) and the state yield (all years),
each including the pooled rows.

  uv run python data/gate/withheld.py
"""
import json, pathlib

import pandas

from qdann import predict, weather

D = weather.DATA_DIR
HERE = pathlib.Path(__file__).parent
MODEL = D / "models" / "qdann_maize_2008_2023_l7_2021.pt"

source = weather.read(D / "source_mpc_maize_2008_2023_l7_2021.parquet")
harm = weather.read(D / "mpc_ext_l7_2021" / "19" / "gcvi_mpc_maize.parquet")
grid = weather.read(D / "extend" / "gridmet_maize_2019_2023.parquet")
nass = pandas.read_parquet(HERE / "nass_ia.parquet")
districts = pandas.read_parquet(D / "district" / "districts.parquet").set_index("fips")["asd"]

feats = harm.merge(grid, on=["fips", "year"], how="inner")
labelled = set(zip(source["fips"], source["year"]))
withheld = feats[[(f, y) not in labelled for f, y in zip(feats["fips"], feats["year"])]].copy()
model, saved = predict.load("maize", MODEL)
withheld["pred"] = predict.apply(model, saved, withheld)[0]
published = nass[nass["fips"].notna()]
withheld["w"] = withheld["fips"].map(published.groupby("fips")["area_ha"].mean())
withheld["asd"] = withheld["fips"].map(districts)
y = lambda d: float(d["production_t"].sum() / d["area_ha"].sum())  # noqa: E731

rows = []
for (year, asd), pool in nass[nass["fips"].isna()].groupby(["year", "asd"]):
    if year < 2019:
        continue
    yr = withheld[withheld["year"] == year]
    members = yr if asd.endswith("-99") else yr[yr["asd"] == asd]
    # a county withheld in NASS but with a published row is not a member: check both ways
    assert not members["fips"].isin(published[published["year"] == year]["fips"]).any()
    if members.empty:
        continue
    year_rows = nass[nass["year"] == year]
    row = {"year": int(year), "pool": asd, "n_counties": len(members),
           "nass_pool": y(pool), "qdann": float((members["pred"] * members["w"]).sum() / members["w"].sum()),
           "state": y(year_rows)}
    if not asd.endswith("-99"):
        row["district"] = y(year_rows[year_rows["asd"] == asd])
    rows.append(row)

out = pandas.DataFrame(rows)
for col in ("qdann", "district", "state"):
    if col in out:
        out[f"{col}_err"] = out[col] - out["nass_pool"]
print(out.round(2).to_string(index=False))
(HERE / "withheld.json").write_text(json.dumps(rows, indent=2))
