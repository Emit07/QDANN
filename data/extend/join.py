"""Join the 2019-2023 MPC harmonics to their labels and weather, and append the result to
the 2008-2018 source table unchanged (sherlock_extend_fit.sbatch).

    python data/extend/join.py <gcvi_mpc_maize.parquet> <out.parquet>

The 2008-2018 rows come from data/source_mpc_maize.parquet as is; only 2019-2023 is joined
here, with the same weather.join that built the original table.
"""

import argparse
import pathlib

import pandas

from qdann import gee, weather

CROP = "maize"
EXTEND = pathlib.Path(__file__).parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gcvi", type=pathlib.Path, help="2019-2023 harmonics from mpc --fit")
    parser.add_argument("out", type=pathlib.Path, help="the 2008-2023 table to write")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; delete it first to rebuild")

    base = weather.read(gee.DATA_DIR / f"source_mpc_{CROP}.parquet")
    new = weather.join(
        labels=weather.read(EXTEND / f"nass_{CROP}_2019_2023.parquet"),
        gcvi=weather.read(args.gcvi),
        weather=weather.read(EXTEND / f"gridmet_{CROP}_2019_2023.parquet"),
        crop=CROP,
    )
    assert list(new.columns) == list(base.columns), (list(new.columns), list(base.columns))
    overlap = set(base["year"]) & set(new["year"])
    assert not overlap, f"years in both tables: {sorted(overlap)}"

    table = pandas.concat([base, new], ignore_index=True)
    table = table.sort_values(["year", "fips"], ignore_index=True)
    assert not table.duplicated(["fips", "year"]).any()
    assert table.notna().all().all(), table.columns[table.isna().any()].tolist()

    table.to_parquet(args.out, index=False)
    print(table.groupby("year").size().to_string())
    print(f"\n{len(base):d} + {len(new):d} = {len(table):d} samples -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
