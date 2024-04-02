import math

import duckdb
from duckdb.typing import DOUBLE
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

import udfs


def calc(points_parquet_filepath: str) -> float:
    pc.register_vector_function(
        lambda ctx, x0, y0, x1, y1: udfs.haversine_dist(x0, y0, x1, y1),
        "haversine_dist",  # name
        {  # doc
            "summary": "calculate haversine distance of two points",
            "description": "calculate haversine distance of two points",
        },
        {
            "x0": pa.float64(),  # input
            "y0": pa.float64(),  # input
            "x1": pa.float64(),  # input
            "y1": pa.float64(),  # input
        },
        pa.float64(),  # output
    )

    def haversine_dist(x0, y0, x1, y1):
        return pc.call_function("haversine_dist", [x0, y0, x1, y1])

    conn = duckdb.connect()
    conn.create_function(
        "haversine_dist",
        haversine_dist,
        [DOUBLE, DOUBLE, DOUBLE, DOUBLE],
        DOUBLE,
        type="arrow",
    )

    sql = f"""
    select avg(haversine_dist(x0, y0, x1, y1)) as avg_dist
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
