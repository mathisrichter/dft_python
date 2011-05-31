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

class KernelMode:
    "Mode of a kernel"

    def __init__(self, amplitude, widths, shifts, kernel):
        self._amplitude = amplitude 
        self._widths = widths
        self._shifts = shifts
        self._kernel = kernel

        dimensionality = self._kernel.get_dimensionality()
        if len(shifts) != len(widths) or len(shifts) != dimensionality:
            print("Error. Number of shift or width values does not match dimensionality of the kernel.")
            
        self._separated_kernel_parts = []

    def get_amplitude(self):
        return self._amplitude

    def get_width(self, dimension_index):
        self.check_dimension_index(dimension_index)
        return self._widths[dimension_index]

    def get_widths(self):
        return self._widths

    def get_shift(self, dimension_index):
        self.check_dimension_index(dimension_index)
        return self._shifts[dimension_index]

    def get_shifts(self):
        return self._shifts

    def get_separated_kernel_part(self, dimension_index):
        self.check_dimension_index(dimension_index)
        return self._separated_kernel_parts[dimension_index]

    def set_amplitude(self, amplitude):
        self._amplitude = amplitude

    def set_width(self, width, dimension_index):
        self.check_dimension_index(dimension_index)
        self._widths[dimension_index] = width

    def set_shifts(self, shift, dimension_index):
        self.check_dimension_index(dimension_index)
        self._shifts = shifts

    def calculate_separated_kernel_parts(self):
        del(self._separated_kernel_parts[:])

        for dimension_index in range(self._kernel.get_dimensionality()):
            kernel_width = self._kernel.get_dimension_size(dimension_index)

            center = (kernel_width / 2.0) + self._shifts[dimension_index]

            kernel_part = numpy.zeros(shape=kernel_width)
            ramp = numpy.linspace(0, kernel_width, kernel_width) 
            for i in range(kernel_width):
                kernel_part[i] = math.exp(-math.pow(ramp[i] - center, 2.0) /       \
                                         (2.0 * math.pow(self._widths[dimension_index], 2.0)))

            # normalize kernel part
            amplitude_sign = math.copysign(1.0, self._amplitude)
            kernel_part *= (amplitude_sign / kernel_part.sum())

            # multiply the first kernel part with the amplitude.
            # when convolving with all separated kernel parts, this will lead
            # to the correct amplitude value for the "whole kernel"
            if (dimension_index == 0):
                kernel_part *= math.fabs(self._amplitude)

            self._separated_kernel_parts.append(kernel_part)

    def check_dimension_index(self, dimension_index):
        dimensionality = self._kernel.get_dimensionality()
        if not (dimension_index >= 0 and dimension_index < dimensionality):
            print("Error. Kernel mode only supports ", dimensionality, "dimensions.")

class Kernel:
    "n-dimensional kernel"
    
    def __init__(self, dimensionality):
        self._dimensionality = dimensionality
        self._dimension_sizes = []
    
    def get_dimension_size(self, dimension_index):
        return self._dimension_sizes[dimension_index]

    def get_dimension_sizes(self):
        return self._dimension_sizes

    def get_dimensionality(self):
        return self._dimensionality
    
    def get_separated_kernel_part(self, dimension_index):
        pass
    

class BoxKernel(Kernel):
    "n-dimensional box kernel"
    
    def __init__(self, dimensionality=1):
        Kernel.__init__(self, 1)
        self._dimensionality = dimensionality
        self._amplitude = 5.0
        self._compute_kernel()
    
    def get_separated_kernel_part(self, dimension_index):
        return self._kernel
        
    def get_amplitude(self):
        return self._amplitude
    
    def set_amplitude(self, amplitude):
        self._amplitude = amplitude        
        self._compute_kernel()

    def _compute_kernel(self):
        self._kernel = numpy.ones(shape=(1)) * self._amplitude
        


class GaussKernel(Kernel):
    "n-dimensional Gauss kernel"

    def __init__(self, dimensionality):
        "Constructor"
        
        Kernel.__init__(self, dimensionality)

        self._modes = []
        self._separated_kernel_parts = []
        self._limit = 0.1

    def add_mode(self, amplitude, widths, shift):
        mode = KernelMode(amplitude, widths, shift, self)
        self._modes.append(mode)

    def get_mode(self, mode_index):
        return self._modes[mode_index]

    def get_separated_kernel_part(self, dimension_index):
        if not (dimension_index >= 0 and dimension_index < self._dimensionality):
            print("Error. Kernel only has", self._dimensionality, "dimensions. You wanted dimension ", dimension_index, ".")
#        print("number of kernel parts: ", len(self._separated_kernel_parts))

        return self._separated_kernel_parts[dimension_index]
 
    def compute_dimension_size(self, dimension_index):
        if not (dimension_index >= 0 and dimension_index < self._dimensionality):
            print("Error. Kernel only supports ", self._dimensionality, " dimensions.")
        
        max_width = 0
        for mode in self._modes:
            mode_width = mode.get_width(dimension_index)
            amplitude = mode.get_amplitude()

            width = 1
            if (mode_width < 10000 and mode_width > 0):
                width =  round(math.sqrt(2.0 * math.pow(mode_width, 2.0) * math.log(math.fabs(amplitude) / self._limit))) + 1
            else:
                print("Error. Selected mode with is not in the proper bounds (0 < width < 10000).")

            max_width = int(max(width, max_width))

        return max_width

    def calculate(self):
        del self._dimension_sizes[:]
        del self._separated_kernel_parts[:]

#        print("calculating kernel")
        for dimension_index in range(self._dimensionality):
            dimension_size = self.compute_dimension_size(dimension_index)
            self._dimension_sizes.append(dimension_size)
            self._separated_kernel_parts.append(numpy.zeros(shape=dimension_size))
#        print("  number of kernel parts: ", len(self._separated_kernel_parts))

        for mode in self._modes:
            mode.calculate_separated_kernel_parts()

            for dimension_index in range(self._dimensionality):
                self._separated_kernel_parts[dimension_index] += mode.get_separated_kernel_part(dimension_index)

    # HACKED
    def recompute_with_parameters(self, amplitude=None, width=None, shift=None):
        widths = None
        shifts = None
#        print("recomputing kernel")

        if (amplitude is None):
            amplitude = self._modes[0].get_amplitude()
        if (width is None):
            widths = self._modes[0].get_widths()
        else:
            widths = [width] * self._dimensionality
        if (shift is None):
            shifts = self._modes[0].get_shifts()
        else:
            shifts = [shift] * self._dimensionality
#        print("  0number of kernel parts: ", len(self._separated_kernel_parts))

#        print("  1number of kernel parts: ", len(self._separated_kernel_parts))
        del self._modes[:]
#        print("  2number of kernel parts: ", len(self._separated_kernel_parts))

        self.add_mode(amplitude, widths, shifts)
#        print("  3number of kernel parts: ", len(self._separated_kernel_parts))
        self.calculate()
#        print("  4number of kernel parts: ", len(self._separated_kernel_parts))

       

