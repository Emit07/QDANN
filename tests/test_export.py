"""The 5' export conserves every published total and lands on the lab's frame."""

import numpy
import pandas
import pyproj
from rasterio.transform import Affine

from qdann import export, mpc


def _stats(rows):
    return pandas.DataFrame(rows, columns=["fips", "asd", "year", "area_ha", "production_t"])


def test_a_pixel_lands_in_the_5_minute_cell_containing_it():
    # one 30 m pixel centred near Ames, Iowa (-93.62, 42.03), on its UTM 15N grid
    crs = pyproj.CRS.from_epsg(32615)
    x, y = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(-93.62, 42.03)
    on = mpc.Grid(crs=crs, transform=Affine(30, 0, x - 15, 0, -30, y + 15), width=1, height=1)
    cells, counts = export.cell_counts(numpy.ones((1, 1), dtype=bool), on)
    row, col = divmod(int(cells[0]), export.FRAME[1])
    assert (row, col) == (int((90 - 42.03) * 12), int((-93.62 + 180) * 12))
    assert counts.tolist() == [1]


def test_published_and_withheld_totals_are_conserved_and_cov_marks_the_gap():
    pixels = {
        "19001": (numpy.array([0, 1]), numpy.array([3, 1])),  # published
        "19003": (numpy.array([1]), numpy.array([2])),  # withheld, district 19-10
        "19005": (numpy.array([2]), numpy.array([4])),  # withheld, district 19-10
        "19007": (numpy.array([3]), numpy.array([5])),  # withheld, no combined row
    }
    stats = _stats(
        [
            ["19001", "19-10", 2018, 100.0, 1000.0],
            [None, "19-10", 2018, 60.0, 540.0],
        ]
    )
    districts = {"19001": "19-10", "19003": "19-10", "19005": "19-10", "19007": "19-20"}
    grids = export.spread(pixels, stats, districts)
    assert all(g.shape == export.FRAME == (2160, 4320) for g in grids.values())
    p, h, cov = (grids[k].ravel() for k in ("P", "H", "cov"))
    assert p.sum() == 1540.0 and h.sum() == 160.0
    # 19001's 1000 t split 3:1; 19-10's 540 t split over 2 + 4 pixels
    numpy.testing.assert_allclose(p[:4], [750.0, 250.0 + 180.0, 360.0, 0.0])
    numpy.testing.assert_allclose(cov[:4], [1.0, 1.0, 1.0, 0.0])
