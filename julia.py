import time
from functools import wraps

from numba import jit
from line_profiler import profile


def time_fn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print(f"@time_fn: {fn.__name__} took {t2 - t1} seconds")
        return result

    return measure_time


@jit(nopython=True)
def calculate_z_serial_purepython(max_iter, zs, c):
    output = [0] * len(zs)
    for i in range(len(zs)):
        n = 0
        z = zs[i]
        while abs(z) < 2 and n < max_iter:
            z = z * z + c
            n += 1
        output[i] = n
    return output


def calc_pure_python(desired_width, max_iterations):
    x1, x2 = -1.8, 1.8
    y1, y2 = -1.8, 1.8
    c_real, c_imag = -0.62772, -0.42193

    xs = [0.0 for _ in range(desired_width)]
    x = x1
    x_step = (x2 - x1) / desired_width
    for i in range(desired_width):
        xs[i] = x
        x += x_step

    ys = [0.0 for _ in range(desired_width)]
    y_step = (y2 - y1) / desired_width
    y = y1
    for i in range(desired_width):
        ys[i] = y
        y += y_step

    c = complex(c_real, c_imag)
    zs = []
    for ycoord in ys:
        for xcoord in xs:
            zs.append(complex(xcoord, ycoord))

    output = calculate_z_serial_purepython(max_iterations, zs, c)
    return output


def main():
    output = calc_pure_python(desired_width=1000, max_iterations=300)
    assert sum(output) == 33219980


if __name__ == "__main__":
    main()
