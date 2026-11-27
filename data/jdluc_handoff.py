"""Step 6: feed QDANN's county yields to jdluc's production pricing, offline.

  uv run --no-sync --project ~/Documents/jdluc python data/jdluc_handoff.py

The emissions frame is TWO_LEVEL_EMISSIONS's shape with made-up hectares: what's checked
is that jdluc accepts the table and prices each county at the QDANN yield.
"""

import pandas

from jdluc.trace import derive_jurisdictional_production_kg

yields = pandas.read_parquet("data/qdann_yields_maize.parquet")
print(yields.to_string())
names = yields.index.to_frame(index=False)
hectares = {admin_id: 1000.0 * (i + 1) for i, admin_id in enumerate(names["admin_id"])}
emissions = (
    pandas.DataFrame.from_records(
        [("PROVINCIAL", "MAIZE", "Iowa", "USA016", 10000.0, 1.0)]
        + [
            ("DISTRICT", "MAIZE", row.jurisdiction_name, row.admin_id, hectares[row.admin_id], 1.0)
            for row in names.itertuples()
        ],
        columns=["admin_level", "crop_name", "jurisdiction_name", "admin_id", "crop_hectares", "emissions_mt"],
    )
    .assign(methodology="JURISDICTIONAL_DIRECT")
    .set_index(["admin_level", "crop_name", "jurisdiction_name", "methodology"])
)
result = derive_jurisdictional_production_kg(emissions=emissions, raw_yields=yields)
print(result[["crop_hectares", "production_kg", "yield_tier"]].to_string())
counties = result.xs("DISTRICT", level="admin_level").droplevel(["crop_name", "methodology"])
for row in names.itertuples():
    got = counties.loc[row.jurisdiction_name]
    assert got["production_kg"] == hectares[row.admin_id] * yields.xs(row.admin_id, level="admin_id")["yield_kg_per_ha"].mean()
    assert got["yield_tier"] == "QDANN"
print(f"OK: jdluc priced {len(names):d} counties at the QDANN yield")
