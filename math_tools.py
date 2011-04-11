import math
import numpy

def product(value_list):
    product = 1    
    for value in value_list:
        product *= value

    return product

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
  
