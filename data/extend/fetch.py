"""Labels and county weather for 2019-2023, the same two legs that built
source_mpc_maize.parquet's 2008-2018 rows (nass.fetch; weather.fetch, Earth Engine gridMET).
Laptop, minutes. Writes new files under data/extend/; skips one that exists.

  uv run python data/extend/fetch.py
"""

import pathlib

import ee

from qdann import env, nass, weather

HERE = pathlib.Path(__file__).parent
YEARS = range(2019, 2024)

labels = HERE / "nass_maize_2019_2023.parquet"
if not labels.exists():
    nass.fetch(crop="maize", years=YEARS, state_fips=("19",)).to_parquet(labels, index=False)
grid = HERE / "gridmet_maize_2019_2023.parquet"
if not grid.exists():
    ee.Initialize(project=env.get("EE_PROJECT"))
    weather.fetch(crop="maize", years=YEARS, state_fips="19").to_parquet(grid, index=False)
for path in (labels, grid):
    t = weather.read(path)
    print(path.name, t.shape, t.groupby("year").size().to_dict(), "NaN cells", int(t.isna().sum().sum()))
