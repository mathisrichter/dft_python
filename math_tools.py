import math
import numpy
import scipy.interpolate

def linear_interpolation_1d(input_array, output_size):
    x = range(len(input_array))
    f = scipy.interpolate.interp1d(x, input_array)

    number_of_steps = complex(0, output_size)
    grid = numpy.mgrid[0:len(input_array)-1:number_of_steps]

    return f(grid)

def linear_interpolation_2d_custom(input_array, output_sizes):
    old_dim0_size = len(input_array)
    tmp = numpy.zeros((old_dim0_size, output_sizes[1]))
    xp0 = range(len(input_array[0]))
    for i in range(old_dim0_size):
        tmp[i] = numpy.interp(numpy.arange(0,
                              len(xp0),
                              float(len(xp0))/float(output_sizes[1])),
                              xp0,
                              input_array[i])

    tmp = tmp.transpose()

    old_dim0_size = len(tmp)
    output = numpy.zeros((output_sizes[1], output_sizes[0]))
    xp0 = range(len(tmp[0]))
    for i in range(old_dim0_size): 
        output[i] = numpy.interp(numpy.arange(0,
                                              len(xp0),
                                              float(len(xp0))/float(output_sizes[0])),
                                              xp0,
                                              tmp[i])
    return output.transpose()



def linear_interpolation_nd(input_array, output_sizes):
    coord_ranges = [range(input_array.shape[i]) for i in range(input_array.ndim)]
    linear_interpolator = scipy.interpolate.LinearNDInterpolator(cartesian(coord_ranges), input_array.flatten())

    coord_ranges_new = [numpy.linspace(0, input_array.shape[i]-1, num=output_sizes[i]) for i in range(input_array.ndim)]
    x_new = cartesian(coord_ranges_new)
    return linear_interpolator(x_new).reshape(output_sizes)

def product(value_list):
    product = 1    
    for value in value_list:
        product *= value

    return product

def sigmoid(x, beta, x0):
    return 1./ (1. + numpy.exp(-beta * (x - x0)))

def gauss_value(position, sigma, shift):
    return math.exp(- math.pow(position - shift, 2.0) / (2 * math.pow(sigma, 2.0)))

def gauss_1d(size, amplitude, sigma, shift):
    gauss = numpy.zeros(size)
    for i in range(size):
        gauss[i] = amplitude * gauss_value(i, sigma, shift)

    return gauss

def gauss_2d(sizes, amplitude, sigmas, shifts):
    gauss = numpy.zeros(sizes)
    for i in range(sizes[0]):
        for j in range(sizes[1]):
            gauss[i][j] = amplitude * (gauss_value(i, sigmas[0], shifts[0]) * gauss_value(j, sigmas[1], shifts[1]))

    return gauss

def gauss_3d(sizes, amplitude, sigmas, shifts):
    gauss = numpy.zeros(sizes)
    for i in range(sizes[0]):
        for j in range(sizes[1]):
            for k in range(sizes[1]):
                gauss[i][j][k] = amplitude * gauss_value(i, sigmas[0], shifts[0]) * gauss_value(j, sigmas[1], shifts[1]) * gauss_value(k, sigmas[2], shifts[2])

    return gauss
  
def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
    """

    arrays = [numpy.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = numpy.prod([x.size for x in arrays])
    if out is None:
        out = numpy.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = numpy.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out


