import math

import duckdb
from duckdb.typing import DOUBLE
import numpy as np
import numba


@numba.jit(nopython=True, nogil=True, parallel=False)
def calc_haversine_dist_py_jit(x0, y0, x1, y1):
    # x -> longitude
    # y -> latitude
    EARTH_RADIUS = 6372.8  # km

    p0_latitude = np.radians(y0)
    p1_latitude = np.radians(y1)

    delta_latitude = np.radians(y0 - y1)
    delta_longitude = np.radians(x0 - x1)

    central_angle_inner = np.square(np.sin(delta_latitude / 2.0)) + np.cos(
        p0_latitude
    ) * np.cos(p1_latitude) * np.square(np.sin(delta_longitude / 2.0))
    central_angle = 2.0 * np.arcsin(np.sqrt(central_angle_inner))

    distance = EARTH_RADIUS * central_angle
    return distance


def calc_haversine_dist_py_nojit(x0, y0, x1, y1):
    # x -> longitude
    # y -> latitude
    EARTH_RADIUS = 6372.8  # km

    p0_latitude = math.radians(y0)
    p1_latitude = math.radians(y1)

    delta_latitude = math.radians(y0 - y1)
    delta_longitude = math.radians(x0 - x1)

    central_angle_inner = math.pow(
        math.sin(delta_latitude / 2.0), 2
    ) + math.cos(p0_latitude) * math.cos(p1_latitude) * math.pow(
        math.sin(delta_longitude / 2.0), 2
    )
    central_angle = 2.0 * math.asin(math.sqrt(central_angle_inner))

    distance = EARTH_RADIUS * central_angle
    return distance


def calc(points_parquet_filepath: str, use_jit_version=False) -> float:
    fn = calc_haversine_dist_py_nojit
    if use_jit_version:
        fn = calc_haversine_dist_py_jit
    conn = duckdb.connect()
    conn.create_function(
        "haversine_dist",
        fn,
        [DOUBLE, DOUBLE, DOUBLE, DOUBLE],
        DOUBLE,
    )
    sql = f"""
    select avg(haversine_dist(x0,y0,x1,y1)) as avg_dist
    from '{points_parquet_filepath}'
    """
    res = conn.sql(sql).fetchall()
    return res[0][0]


def main():
    n = "10"
    points_parquet_filepath = f"data/points_{n}.parquet"
    res = calc(points_parquet_filepath)
    print(res)


if __name__ == "__main__":
    main()
