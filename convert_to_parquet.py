import duckdb

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    n = "10M"
    input_json = f"data/{n}/points.json"
    output_parquet_filepath = f"points_{n}.parquet"
    sql = f"""
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
    select
        p.x0, p.y0, p.x1, p.y1
    from points limit 5
    """
    conn = duckdb.connect()
    tbl = conn.sql(sql).arrow()
    pq.write_table(
        tbl,
        output_parquet_filepath,
        compression="ZSTD",
    )


if __name__ == "__main__":
    main()
