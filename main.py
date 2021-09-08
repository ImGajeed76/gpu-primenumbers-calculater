
import numba
import numpy as np
from numba import cuda
from numba_timer import cuda_timer

max = 1000000
split_up = 5

@cuda.jit
def is_prime(r, d):
    """

    :param r:
    :param d:
    """
    x, y = cuda.grid(2)
    d_x = cuda.gridDim.x
    d_y = cuda.gridDim.y
    dim = d_x * d_y

    for j in range(d[0]):
        n = ((y * x) - (d_x - x)) + (dim * j)
        for i in range(n - 2):
            if (n % (i + 2)) == 0:
                r[n] = 0
                break
            else:
                r[n] = n


@numba.jit(nopython=True)
def get_blocks(depth, num):
    """

    :param depth:
    :param num:
    :return:
    """
    result = 1

    for i in range(depth):
        if num % (i + 1) == 0 and (i + 1) > result:
            result = (i + 1)

    return result


@numba.jit(nopython=True)
def deleteZeors(array):
    """

    :param array:
    :return:
    """
    out = []

    for i in array:
        if i != 0:
            out.append(int(i))

    return out


def getPrimeNumbers(depth):
    """

    :param depth:
    :return:
    """
    d_r = cuda.to_device(np.zeros(depth))
    divider = get_blocks(split_up, depth)
    blocks = get_blocks(int(depth / 100), depth / divider)
    d_d = cuda.to_device(np.array([divider]))
    is_prime[int(blocks), int((depth / divider) / blocks)](d_r, d_d)
    r = d_r.copy_to_host()

    primeNumbers = deleteZeors(r)

    return primeNumbers


timer = cuda_timer.Timer()

timer.start()
print(getPrimeNumbers(max))
timer.stop()

print()
print("Calculated the prime-numbers from 0 to " + str(max) + " in")

duration = timer.elapsed()
print(str(duration) + " ms")
duration_s = timer.elapsed_seconds()
print(str(duration_s) + " s")
