import numpy as np
from scipy.signal import correlate2d
from numba import jit
from numba import cuda


@cuda.jit
def gpu_convolve2d(input_plane, kernel, output):
    # Get thread position in the grid
    i, j = cuda.grid(2)

    # Dimensions of kernel
    kernel_height, kernel_width = kernel.shape

    # Dimensions of input and output
    input_height, input_width = input_plane.shape
    output_height, output_width = output.shape

    # Perform convolution only if within output bounds
    if i < output_height and j < output_width:
        value = 0.0
        for ki in range(kernel_height):
            for kj in range(kernel_width):
                value += input_plane[i + ki, j + kj] * kernel[ki, kj]
        output[i, j] = value

# Wrapper function


def convolve2d_gpu(input_plane, kernel):
    kernel_height, kernel_width = kernel.shape
    output_height = input_plane.shape[0] - kernel_height + 1
    output_width = input_plane.shape[1] - kernel_width + 1

    # Allocate arrays on the GPU
    d_input_plane = cuda.to_device(input_plane)
    d_kernel = cuda.to_device(kernel)
    d_output = cuda.device_array(
        (output_height, output_width), dtype=np.float32)

    # Define grid and block size
    threads_per_block = (16, 16)  # Tune based on your GPU's capabilities
    blocks_per_grid_x = (
        output_height + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (
        output_width + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Launch the kernel
    gpu_convolve2d[blocks_per_grid, threads_per_block](
        d_input_plane, d_kernel, d_output)

    # Copy the result back to the host
    return d_output.copy_to_host()


def convolve2d_np(input_plane, kernel):
    kernel_height, kernel_width = kernel.shape
    # Generate sliding windows of the input
    windows = np.lib.stride_tricks.sliding_window_view(
        input_plane, (kernel_height, kernel_width)
    )
    # Perform element-wise multiplication and sum over the last two axes
    output = np.einsum('ijkl,kl->ij', windows, kernel)
    return output


@jit(nopython=True)
def convolve2d_numba(input_plane, kernel):
    kernel_height, kernel_width = kernel.shape
    output_height = input_plane.shape[0] - kernel_height + 1
    output_width = input_plane.shape[1] - kernel_width + 1
    output = np.zeros((output_height, output_width), dtype=np.float64)

    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = np.sum(
                input_plane[i:i+kernel_height, j:j+kernel_width] * kernel)
    return output


def convolve2d(input_plane, kernel):
    # return convolve2d_gpu(input_plane, kernel)
    return convolve2d_numba(input_plane, kernel)
    # return correlate2d(input_plane, kernel, mode='valid')
