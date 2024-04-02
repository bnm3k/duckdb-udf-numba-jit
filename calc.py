import os
import time
import argparse


def get_calc_fn(method):
    if method == "udf_py":
        from method_udf_py import calc

        return calc
    elif method == "duckdb_sql":
        from method_duckdb_sql import calc

        return calc
    elif method == "udf_numba":
        from method_udf_numba import calc

        return calc
    elif method == "udf_rs":
        from method_udf_numba import calc

        return calc
    return None


def main():
    print(f"pid={os.getpid()}")
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="file containing points")
    parser.add_argument("-m", "--method", help=" method to use for calc")
    args = parser.parse_args()

    input_filepath = args.file
    if input_filepath is None:
        n = "10"
        input_filepath = f"data/points_{n}.parquet"

    method = args.method
    if method is None:
        print("Error: method is empty. Provide method arg")
        exit(1)

    calc = get_calc_fn(method)
    if calc is None:
        print(f"Invalid method: {method}")
        exit(1)

    start = time.time()
    res = calc(input_filepath)
    end = time.time()
    print(res)
    print(f"Time taken: {end - start}")


if __name__ == "__main__":
    main()
