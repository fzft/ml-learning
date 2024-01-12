from numba import cuda
import numpy as np
from timeit import default_timer as timer


@cuda.jit
def fill_array_with_gpu(a):
    pos = cuda.grid(1)  # Compute thread position
    if pos < a.size:  # Check array bounds
        a[pos] += 1


if __name__ == '__main__':
    r = 1000000000
    a = np.ones(r, dtype=np.float32)  # Ensure data type is supported by CUDA

    threads_per_block = 1024  # Adjust as needed
    blocks_per_grid = (a.size + (threads_per_block - 1)) // threads_per_block

    start = timer()
    for _ in range(100):
        fill_array_with_gpu[blocks_per_grid, threads_per_block](a)
    print("with GPU:", timer() - start)
