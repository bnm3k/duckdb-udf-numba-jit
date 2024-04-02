import threading
import types

import duckdb
from duckdb.typing import *


import pyarrow as pa
import pyarrow.compute as pc

import numpy as np
import numba
from numba.experimental import jitclass

spec = [
    ("missing", numba.uint8[:]),
    ("data", numba.float64[:]),
]


@jitclass(spec)
class NumbaFloat64Array:
    def __init__(self, missing, data):
        self.missing = missing
        self.data = data


def _make(cls, pa_arr):
    assert isinstance(pa_arr, pa.FloatingPointArray)
    buffers = pa_arr.buffers()
    return cls(
        np.asarray(buffers[0]).view(np.uint8),
        np.asarray(buffers[1]).view(np.float64),
    )


NumbaFloat64Array.make = types.MethodType(_make, NumbaFloat64Array)


@numba.jit(nopython=True, nogil=True, parallel=False)
def _add(xs, ys, out, len_):
    for i in range(len_):
        out[i] = xs[i] + ys[i]


threads = set()


def add_cols(xs, ys):
    tid = threading.get_native_id()
    threads.add(tid)
    len_ = len(xs)
    out = np.empty((len_,))
    _add(xs.to_numpy(), ys.to_numpy(), out, len_)
    return pa.array(out)


def main():
    conn = duckdb.connect()
    conn.create_function(
        "add_cols", add_cols, [DOUBLE, DOUBLE], DOUBLE, type="arrow"
    )
    conn.sql("create table tbl(xs double, ys double)")
    n = 1000000
    conn.sql(
        f"""
        insert into tbl(xs,ys)
        select range::double, range::double from range(1,{n+1})
    """
    )
    conn.sql(
        f"""select sum(res) / {n} as s from (
        select add_cols(xs, ys) as res from tbl)"""
    ).show()
    print(threads)


if __name__ == "__main__":
    main()
