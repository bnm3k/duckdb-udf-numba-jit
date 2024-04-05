import time
import math

import numpy as np
import pyarrow as pa
import duckdb
from duckdb.typing import DOUBLE
from numba import jit, vectorize, float64


@jit(nopython=True, nogil=True, parallel=False)
def _calc_haversine_dist_vectorized(x0, y0, x1, y1, out, len_):
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


def _calc_haversine_dist(x0, y0, x1, y1):
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

_calc_haversine_dist_vec = vectorize(spec, nopython=True, target="cpu")(
    _calc_haversine_dist
)

_calc_haversine_dist_cuda = vectorize(spec, target="cuda")(_calc_haversine_dist)

_calc_haversine_dist_py_jit = jit(nopython=True, nogil=True, parallel=False)(
    _calc_haversine_dist
)


def get_calc(udf_type):
    fn = None
    type_ = "arrow"
    if udf_type == "vec_nojit" or udf_type == "vec_jit":
        _calc = _calc_haversine_dist_vectorized
        if udf_type == "vec_jit":
            _calc = jit(nopython=True, nogil=True, parallel=False)(
                _calc_haversine_dist_vectorized
            )

        def fn_numba(x0, y0, x1, y1):
            len_ = len(x0)
            out = np.empty((len_,))
            vs = tuple(v.to_numpy() for v in (x0, y0, x1, y1))
            _calc(*vs, out=out, len_=len_)
            return pa.array(out)

        fn = fn_numba
    elif udf_type == "cuda":
        fn = lambda x0, y0, x1, y1: pa.array(
            _calc_haversine_dist_cuda(
                *tuple(v.to_numpy() for v in (x0, y0, x1, y1))
            )
        )
    elif udf_type == "py" or udf_type == "py_jit":
        fn = _calc_haversine_dist
        if udf_type == "py_jit":
            fn = _calc_haversine_dist_py_jit
        type_ = "native"
    elif udf_type == "numba_vectorize":

        def fn_vec(x0, y0, x1, y1):
            out = _calc_haversine_dist_vec(
                *tuple(v.to_numpy() for v in (x0, y0, x1, y1))
            )
            return pa.array(out)

        fn = fn_vec

    if fn is None:
        return None

    def calc(points_parquet_filepath: str) -> float:
        conn = duckdb.connect()
        conn.create_function(
            "haversine_dist",
            fn,
            [DOUBLE, DOUBLE, DOUBLE, DOUBLE],
            DOUBLE,
            type=type_,
        )
        sql = f"""
        select avg(haversine_dist(x0,y0,x1,y1)) as avg_dist
        from '{points_parquet_filepath}'
        """
        res = conn.sql(sql).fetchall()
        return res[0][0]

    return calc


def main():
    points_parquet_filepath = f"data/points_10M.parquet"
    udf_type = "numba"
    calc = get_calc(udf_type)
    assert calc is not None
    start = time.time()
    res = calc(points_parquet_filepath)
    end = time.time()
    print(res)
    print(f"took: {end - start}")


if __name__ == "__main__":
    main()
