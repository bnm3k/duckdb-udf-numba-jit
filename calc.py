import os
import time
import argparse


def get_calc_fn(method):
    udf_prefix = "udf_"
    if method.startswith(udf_prefix):
        udf_type = method[len(udf_prefix) :]
        if udf_type == "rs":
            from method_udf_rs import calc

            return calc
        else:  # udf_numba, udf_cuda, udf_vec, "udf_py", "udf_py_jit"
            from method_udf import get_calc

            calc = get_calc(udf_type)

            return calc
    elif method == "duckdb_sql":
        from method_duckdb_sql import calc

        return calc
    else:
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

    # start = time.time()
    res = calc(input_filepath)
    # end = time.time()
    # print(f"Time taken: {end - start}")
    print(res)


if __name__ == "__main__":
    main()
