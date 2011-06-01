import math
import random
from scipy import ndimage
import numpy
import copy

def convolve(input, kernel):
    convolution_result = copy.copy(input)
    for dimension_index in range(kernel.get_dimensionality()):
        ndimage.convolve1d(convolution_result, \
                           kernel.get_separated_kernel_part(dimension_index), \
                           axis = dimension_index, \
                           output = convolution_result, \
                           mode = 'wrap')

    return convolution_result


class Kernel:
    "n-dimensional kernel"

    def __init__(self, amplitude):
        self._dimensionality = None
        self._amplitude = amplitude

    def get_amplitude(self):
        return self._amplitude

    def get_separated_kernel_part(self, dimension_index):
        self._check_dimension_index(dimension_index)
        return self._separated_kernel_parts[dimension_index]

    def get_separated_kernel_parts(self):
        return self._separated_kernel_parts

    def get_dimension_size(self, dimension_index):
        return self._dimension_sizes[dimension_index]

    def get_dimension_sizes(self):
        return self._dimension_sizes

    def get_dimensionality(self):
        return self._dimensionality

    def set_amplitude(self, amplitude):
        self._amplitude = amplitude
        self._calculate_kernel()

    def _check_dimension_index(self, dimension_index):
        if not (dimension_index >= 0 and dimension_index < self._dimensionality):
            print("Error. Kernel only has", self._dimensionality, "dimensions.")

    def _calculate_kernel(self):
        pass



class GaussKernel(Kernel):
    "n-dimensional Gauss kernel"

    def __init__(self, amplitude, widths, shifts=None):
        Kernel.__init__(self, amplitude)
        self._widths = widths
        self._shifts = None
        self._dimensionality = len(self._widths)
        self._dimension_sizes = None
        self._limit = 0.1
        self._separated_kernel_parts = None

        if (shifts is None):
            self._shifts = [0.0] * self._dimensionality
        else:
            if (len(self._widths) != len(self._shifts)):
                print("Error. Number of shift and width values does not match.")

        self._calculate_separated_kernel_parts()

    def get_width(self, dimension_index):
        self._check_dimension_index(dimension_index)
        return self._widths[dimension_index]

    def get_widths(self):
        return self._widths

    def get_shift(self, dimension_index):
        self._check_dimension_index(dimension_index)
        return self._shifts[dimension_index]

    def get_shifts(self):
        return self._shifts
    
    def set_width(self, width, dimension_index):
        self._check_dimension_index(dimension_index)
        self._widths[dimension_index] = width
        self._calculate_seperated_kernel_parts()

    def set_shift(self, shift, dimension_index):
        self._check_dimension_index(dimension_index)
        self._shifts = shifts
        self._calculate_seperated_kernel_parts()

    def _calculate_dimension_size(self, dimension_index):
        if not (dimension_index >= 0 and dimension_index < self._dimensionality):
            print("Error. Kernel only supports ", self._dimensionality, " dimensions.")
        
        dimension_width = self._widths[dimension_index]

        if (dimension_width < 10000 and dimension_width > 0):
            dimension_size = int(round(math.sqrt(2.0 * math.pow(dimension_width, 2.0) \
                                 * math.log(math.fabs(self._amplitude) / self._limit))) + 1)
        else:
            print("Error. Selected mode with is not in the proper bounds (0 < width < 10000).")

        if (dimension_size % 2 == 0):
            dimension_size += 1

        return dimension_size

    def _calculate_kernel(self):
        self._calculate_separated_kernel_parts()

    def _calculate_separated_kernel_parts(self):
        if (self._separated_kernel_parts is not None):
            del(self._separated_kernel_parts[:])
        else:
            self._separated_kernel_parts = []

        for dimension_index in range(self._dimensionality):
            dimension_size = self._calculate_dimension_size(dimension_index)

            center = (dimension_size / 2.0) + self._shifts[dimension_index]

            kernel_part = numpy.zeros(shape=dimension_size)
            ramp = numpy.linspace(0, dimension_size, dimension_size) 
            for i in range(dimension_size):
                kernel_part[i] = math.exp(-math.pow(ramp[i] - center, 2.0) / \
                                         (2.0 * math.pow(self._widths[dimension_index], 2.0)))

            # normalize kernel part
            kernel_part *= 1.0 / kernel_part.sum()

            # multiply the first kernel part with the amplitude.
            # when convolving with all separated kernel parts, this will lead
            # to the correct amplitude value for the "whole kernel"
            if (dimension_index == 0):
                kernel_part *= self._amplitude

            self._separated_kernel_parts.append(kernel_part)

class BoxKernel(Kernel):
    "n-dimensional box kernel"
    
    def __init__(self, amplitude = 5.0):
        Kernel.__init__(self, amplitude)
        self._dimensionality = 1
        self._calculate_kernel()
    
    def get_separated_kernel_part(self, dimension_index):
        return self._kernel
        
    def _calculate_kernel(self):
        self._kernel = numpy.ones(shape=(1)) * self._amplitude
