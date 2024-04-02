import os
import mmap

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import orjson


def use_duckdb():
    n = "10M"
    input_json = f"data/{n}/points.json"
    output_parquet_filepath = f"points_{n}.parquet"
    sql = f"""
    create temporary table points as (
        with json_rows as (
            select
               unnest(from_json(pairs, '["JSON"]')) as row
            from read_json(
                '{input_json}',
                format='unstructured',
                columns={{"pairs":"VARCHAR"}},
                maximum_object_size=1200000000)
            as j
        ),
        points as (
            select
                json_transform_strict(
                    row,
                    '{{"x0":"DOUBLE","y0":"DOUBLE","x1":"DOUBLE","y1":"DOUBLE"}}'
                ) as p
            from json_rows as r
        )
        select p.x0, p.y0, p.x1, p.y1 from points
    )
    """
    conn = duckdb.connect()
    conn.execute(sql)
    conn.execute(
        f"""copy
            (select x0,y0,x1,y1 from points)
            to '{output_parquet_filepath}'
            (format 'parquet', codec 'zstd')
            """
    )
    # tbl = conn.sql(sql).arrow()


def json_to_parquet(json, output_path):
    pairs = json["pairs"]
    len_ = len(pairs)
    x0 = np.empty((len_,), dtype=np.float64)
    y0 = np.empty((len_,), dtype=np.float64)
    x1 = np.empty((len_,), dtype=np.float64)
    y1 = np.empty((len_,), dtype=np.float64)
    for i, row in enumerate(pairs):
        x0[i] = row["x0"]
        y0[i] = row["y0"]
        x1[i] = row["x1"]
        y1[i] = row["y1"]

    tbl = pa.Table.from_arrays(
        list(pa.array(arr) for arr in [x0, y0, x1, y1]),
        ["x0", "y0", "x1", "y1"],
    )
    pq.write_table(
        tbl,
        output_path,
        compression="zstd",
    )


def main():
    n = "10M"
    input_json = f"data/{n}/points.json"
    output_path = f"points_{n}.parquet"
    with open(input_json, "r+b") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            m = memoryview(mm)
            parsed = orjson.loads(m)
            json_to_parquet(parsed, output_path)
            m.release()


if __name__ == "__main__":
    main()
