import duckdb


def calc(points_parquet_filepath: str) -> float:
    sql = f"""
    with distances as (
        select
            2 * 6335
                * asin(sqrt(
                    pow(sin((radians(y1) - radians(y0)) / 2), 2)
                    + cos(radians(y0))
                    * cos(radians(y1))
                    * pow(sin((radians(x1) - radians(x0)) / 2), 2)
                )) as dist
        from '{points_parquet_filepath}'
    )
    select
        avg(dist) as avg_dist
    from distances
    """
    res = duckdb.sql(sql).fetchall()
    return res[0][0]


def main():
    points_parquet_filepath = f"data/points_10M.parquet"
    res = calc(points_parquet_filepath)
    print(res)


if __name__ == "__main__":
    main()
