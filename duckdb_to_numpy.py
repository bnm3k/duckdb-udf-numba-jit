import math
import time

import duckdb
import numpy as np
from numba import float64, vectorize, jit, config, threading_layer
import pyarrow.parquet as pq

from method_udf import _calc_haversine_dist


spec = [float64(float64, float64, float64, float64)]
get_dist_cuda = vectorize(spec, target="cuda")(_calc_haversine_dist)
get_dist_parallel = vectorize(spec, nopython=True, target="parallel")(
    _calc_haversine_dist
)


@jit(nopython=True, nogil=True)
def get_avg(nums):
    return np.mean(nums)


def calc(args, get_dist) -> float:
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
    use_cuda = True
    input_filepath = f"data/points_10M.parquet"

    if use_cuda:
        print("cuda")
        get_dist = get_dist_cuda
    else:
        print(f"Threading layer: {threading_layer()}")
        get_dist = get_dist_parallel
    # start = time.time()
    res = calc(read_args(input_filepath, "duckdb"), get_dist)
    print(res)
    # end = time.time()
    # print(f"took: {end - start}")


if __name__ == "__main__":
    main()
