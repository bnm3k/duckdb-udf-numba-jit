import time
from numba import jit


def calculate_z_serial_purepython(max_iter, zs, cs):
    output = [0] * len(zs)
    for i in range(len(zs)):
        n = 0
        z = zs[i]
        c = cs[i]
        while abs(z) < 2 and n < max_iter:
            z = z * z + c
            n += 1
        output[i] = n
    return output


def calc_pure_python(desired_width, max_iterations):
    x1, x2 = -1.8, 1.8
    y1, y2 = -1.8, 1.8
    c_real, c_imag = -0.62772, -0.42193
    # create a list of complex coordinates and complex parameters, build a
    # julia set
    x_step = (x2 - x1) / desired_width
    y_step = (y1 - y2) / desired_width
    x = []
    y = []
    ycoord = y2
    while ycoord > y1:
        y.append(ycoord)
        ycoord += y_step
    xcoord = x1
    while xcoord < x2:
        x.append(xcoord)
        xcoord += x_step

    zs = []
    cs = []
    for ycoord in y:
        for xcoord in x:
            zs.append(complex(xcoord, ycoord))
            cs.append(complex(c_real, c_imag))

    print("Length of x:", len(x))
    print("Total elements:", len(zs))
    start_time = time.time()
    output = calculate_z_serial_purepython(max_iterations, zs, cs)
    end_time = time.time()
    time_taken_secs = end_time - start_time
    print(
        f"{calculate_z_serial_purepython.__name__} took {time_taken_secs} seconds"
    )
    print(100 * sum(1 for n in output if n > 0) / sum(1 for n in output))
    assert sum(output) == 33219980


def main():
    calc_pure_python(desired_width=1000, max_iterations=300)


if __name__ == "__main__":
    main()
