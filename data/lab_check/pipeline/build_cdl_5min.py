"""USDA Cropland Data Layer (30 m, Albers) -> per-crop hectares on the
5-arcmin grid, for the explorer's US 'improved crop map' (__cdl).

    .venv/bin/python pipeline/build_cdl_5min.py [YEAR]

No argument = 2023, the override's own grid (cdl_5min_ha.npz). Any other
year writes cdl_{YEAR}_5min_ha.npz, an epoch of the mask's time dimension
(the zip from NASS is extracted beside itself if the .tif is absent).

CDL is in CONUS Albers (EPSG:5070); exact per-pixel reprojection of
~10^10 pixels is pointless at a 5' target, so pixels are counted into
32x32 (960 m) blocks in Albers space and each block's center is
transformed to lat/lon and binned - sub-km placement error against a
9 km target grid.
"""
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
NY, NX = 2160, 4320
RES5 = 180.0 / NY
BLK = 32
PX_HA = 0.09          # 30 m x 30 m

CDL_TO_SPAM = {
    "MAIZ": [1], "COTT": [2], "RICE": [3], "SORG": [4], "SOYB": [5],
    "GROU": [10], "BARL": [21], "WHEA": [22, 23, 24], "RAPE": [31],
    "BEAN": [42], "POTA": [43], "SUGC": [45],
}


def main():
    t0 = time.time()
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2023
    d = ROOT / "data/raw/crop_masks/cdl"
    f = d / f"{year}_30m_cdls.tif"
    if not f.exists():
        with zipfile.ZipFile(d / f"{year}_30m_cdls.zip") as zf:
            tifs = [n for n in zf.namelist() if n.lower().endswith(".tif")]
            zf.extract(tifs[0], d)
            (d / tifs[0]).rename(f)
    out_path = ROOT / ("data/processed/cdl_5min_ha.npz" if year == 2023
                       else f"data/processed/cdl_{year}_5min_ha.npz")
    out = {c: np.zeros((NY, NX), dtype=np.float64) for c in CDL_TO_SPAM}
    codes = {code: c for c, lst in CDL_TO_SPAM.items() for code in lst}
    code_arr = np.zeros(256, dtype=np.int16) - 1
    crop_ids = list(CDL_TO_SPAM)
    for code, c in codes.items():
        code_arr[code] = crop_ids.index(c)

    with rasterio.open(f) as src:
        tr = src.transform
        transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        H, W = src.height, src.width
        wb, hb = W // BLK, None
        step = 1024  # was 4096; a multiple of 32, so the blocks and result are unchanged
        for r0 in range(0, H, step):
            rows = min(step, H - r0)
            a = src.read(1, window=((r0, r0 + rows), (0, W)))
            cls = code_arr[a]                       # (-1 = not a target crop)
            rb, wb_ = rows // BLK, W // BLK
            cls = cls[:rb * BLK, :wb_ * BLK].reshape(rb, BLK, wb_, BLK)
            # per-block, per-crop counts
            for k in range(len(crop_ids)):
                cnt = (cls == k).sum(axis=(1, 3)).astype(np.float64)
                nz = np.nonzero(cnt)
                if nz[0].size == 0:
                    continue
                # block centers in Albers -> lat/lon
                xs = tr.c + (nz[1] * BLK + BLK / 2) * tr.a
                ys = tr.f + ((r0 + nz[0] * BLK) + BLK / 2) * tr.e
                lon, lat = transformer.transform(xs, ys)
                r5 = np.clip(((90.0 - np.asarray(lat)) / RES5).astype(np.int64), 0, NY - 1)
                c5 = np.clip(((np.asarray(lon) + 180.0) / RES5).astype(np.int64), 0, NX - 1)
                np.add.at(out[crop_ids[k]], (r5, c5), cnt[nz] * PX_HA)
            if (r0 // step) % 8 == 0:
                print(f"  [{time.time()-t0:.0f}s] rows {r0}/{H}", flush=True)

    tot = {c: g.sum() / 1e6 for c, g in out.items()}
    print("Mha by crop:", {c: round(v, 2) for c, v in tot.items()})
    np.savez_compressed(out_path,
                        **{c: g.astype(np.float32) for c, g in out.items()})
    print(f"done ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
