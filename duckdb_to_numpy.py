import time

import duckdb
import numpy as np
from numba import float64, vectorize, jit
import pyarrow.parquet as pq

spec = [float64(float64, float64, float64, float64)]


@vectorize(spec, nopython=True, target="parallel")
def get_dist(x0, y0, x1, y1):
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


@jit(nopython=True, nogil=True, parallel=True)
def get_avg(nums):
    return np.mean(nums)


def calc(args) -> float:
    dists = get_dist(*args)
    avg = get_avg(dists)
    return avg


def read_args(points_parquet_filepath: str, reader):
    if reader == "duckdb":
        sql = f"""
        select x0,y0,x1,y1
        from '{points_parquet_filepath}'
        """
        as_np = duckdb.sql(sql).fetchnumpy()
        args = tuple(as_np[k] for k in ("x0", "y0", "x1", "y1"))
        return args
    elif reader == "pyarrow":
        tbl = pq.read_table(points_parquet_filepath)
        args = tuple(tbl[c].to_numpy() for c in ("x0", "y0", "x1", "y1"))
        return args
    else:
        raise Exception(f"Invalud reader: '{reader}'")


def main():
    input_filepath = f"data/points_10M.parquet"
    start = time.time()
    res = calc(read_args(input_filepath, "duckdb"))
    end = time.time()
    print(res)
    print(f"took: {end - start}")


if __name__ == "__main__":
    main()
