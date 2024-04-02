import duckdb
from duckdb.typing import *


import pyarrow as pa

import numpy as np
import numba


@numba.jit(nopython=True, nogil=True, parallel=False)
def _calc_haversine_dist(x0, y0, x1, y1, out, len_):
    # x -> longitude
    # y -> latitude
    EARTH_RADIUS = 6372.8  # km

    for i in range(len_):
        p0_latitude = np.radians(y0[i])
        p1_latitude = np.radians(y1[i])

        delta_latitude = np.radians(y0[i] - y1[i])
        delta_longitude = np.radians(x0[i] - x1[i])

        central_angle_inner = np.square(np.sin(delta_latitude / 2.0)) + np.cos(
            p0_latitude
        ) * np.cos(p1_latitude) * np.square(np.sin(delta_longitude / 2.0))
        central_angle = 2.0 * np.arcsin(np.sqrt(central_angle_inner))

        distance = EARTH_RADIUS * central_angle
        out[i] = distance


def calc_haversine_dist_numba(x0, y0, x1, y1):
    len_ = len(x0)
    out = np.empty((len_,))
    _calc_haversine_dist(
        x0.to_numpy(), y0.to_numpy(), x1.to_numpy(), y1.to_numpy(), out, len_
    )
    return pa.array(out)


def calc(points_parquet_filepath: str) -> float:
    conn = duckdb.connect()
    conn.create_function(
        "haversine_dist",
        calc_haversine_dist_numba,
        [DOUBLE, DOUBLE, DOUBLE, DOUBLE],
        DOUBLE,
        type="arrow",
    )
    sql = f"""
    select avg(haversine_dist(x0,y0,x1,y1)) as avg_dist
    from '{points_parquet_filepath}'
    """
    res = conn.sql(sql).fetchall()
    return res[0][0]


def main():
    n = "10"
    points_parquet_filepath = f"points_{n}.parquet"
    res = calc(points_parquet_filepath)
    print(res)


if __name__ == "__main__":
    main()
