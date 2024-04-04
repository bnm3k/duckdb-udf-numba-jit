import math
import time

import duckdb
import numpy as np
from numba import float64, vectorize, jit
import pyarrow.parquet as pq


def _get_dist(x0, y0, x1, y1):
    # x -> longitude
    # y -> latitude
    EARTH_RADIUS = 6372.8  # km

    p0_latitude = math.radians(y0)
    p1_latitude = math.radians(y1)

    delta_latitude = math.radians(y0 - y1)
    delta_longitude = math.radians(x0 - x1)

    central_angle_inner = (math.sin(delta_latitude / 2.0)) ** 2 + math.cos(
        p0_latitude
    ) * math.cos(p1_latitude) * (math.sin(delta_longitude / 2.0) ** 2)

    central_angle = 2.0 * math.asin(math.sqrt(central_angle_inner))

    distance = EARTH_RADIUS * central_angle
    return distance


spec = [float64(float64, float64, float64, float64)]
get_dist_cuda = vectorize(spec, target="cuda")(_get_dist)
get_dist_parallel = vectorize(spec, nopython=True, target="parallel")(_get_dist)


@jit(nopython=True, nogil=True, parallel=True)
def get_avg(nums):
    return np.mean(nums)


def calc(args) -> float:
    dists = get_dist_cuda(*args)
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
