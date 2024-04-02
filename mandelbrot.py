from math import ceil

import numpy as np
import cupy as cp
from numba import cuda
from PIL import Image
import matplotlib.cm


@cuda.jit
def compute_all_points(start_x, start_y, end_x, end_y, size, img_array):
    x, y = cuda.grid(2)
    if x >= img_array.shape[0] or y >= img_array.shape[1]:
        return
    mandel_x = (end_x - start_x) * (x / size) + start_x
    mandel_y = (end_y - start_y) * (y / size) + start_y
    c = complex(mandel_x, mandel_y)
    i = -1
    z = complex(0, 0)
    while abs(z) < 2:
        i += 1
        if i == 255:
            break
        z = z**2 + c
    img_array[y, x] = i


def denormalize(palette):
    return [tuple(int(channel * 255) for channel in color) for color in palette]


def main():
    size = 2000
    start_x, start_y = -2.5, -2.3
    end_x, end_y = 1.5, 2.3

    threads_per_block_2d = 16, 16
    blocks_per_grid_2d = ceil(size / 16), ceil(size / 16)
    cp_img_array = cp.empty((size, size), dtype=cp.uint8)
    compute_all_points[blocks_per_grid_2d, threads_per_block_2d](
        start_x, start_y, end_x, end_y, size, cp_img_array
    )

    res = cp.asnumpy(cp_img_array)
    img = Image.fromarray(res, mode="P")
    colormap = matplotlib.cm.get_cmap("twilight").colors
    palette = denormalize(colormap)
    img.putpalette(palette)

    img = img.convert("HSV")
    img.show()
    # Generate random values between 0 and 255 into a numpy array
    # random_values = np.random.randint(0, 256, size=(300, 300), dtype=np.uint8)

    # # Convert the numpy array to Pillow image
    # img = Image.fromarray(random_values, mode="P")

    # # Display the image

    # img.convert("HSV").show()


if __name__ == "__main__":
    main()
