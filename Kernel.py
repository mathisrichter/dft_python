import math
import random
from scipy import ndimage
import numpy
import copy

def sigmoid(x, beta, x0):
    return 1./ (1. + math.exp(-beta * (x - x0)))

def convolve(input, kernel):
    convolution_result = copy.copy(input)
    for dimension_index in xrange(kernel.get_dimensionality()):
        ndimage.convolve1d(convolution_result, kernel.get_separated_kernel_parts(dimension_index), axis=dimension_index, output=convolution_result, mode='wrap')

    return convolution_result

class KernelMode:
    "Mode of a kernel"

    def __init__(self, amplitude, steepnesses, shifts, kernel):
        self._amplitude = amplitude 
        self._steepnesses = steepnesses
        self._shifts = shifts
        self._kernel = kernel

        dimensionality = self._kernel.get_dimensionality()
        if len(shifts) != len(steepnesses) or len(shifts) != dimensionality:
            print("Error. Number of shift or steepness values does not match dimensionality of the kernel.")
            
        self._separated_kernel_parts = []

    def get_amplitude(self):
        return self._amplitude

    def get_steepness(self, dimension_index):
        self.check_dimension_index(dimension_index)
        return self._steepnesses[dimension_index]

    def get_shifts(self, dimension_index):
        self.check_dimension_index(dimension_index)
        return self._shifts[dimension_index]

    def get_separated_kernel_part(self, dimension_index):
        self.check_dimension_index(dimension_index)
        return self._separated_kernel_parts[dimension_index]

    def set_amplitude(self, amplitude):
        self._amplitude = amplitude

    def set_steepness(self, steepness, dimension_index):
        self.check_dimension_index(dimension_index)
        self._steepnesses[dimension_index] = steepness

    def set_shifts(self, shift, dimension_index):
        self.check_dimension_index(dimension_index)
        self._shifts = shifts

    def calculate_separated_kernel_parts(self):
        del(self._separated_kernel_parts[:])

        for dimension_index in xrange(self._kernel.get_dimensionality()):
            kernel_width = self._kernel.get_dimension_size(dimension_index)

            center_index = math.floor(kernel_width / 2) + round(self._shifts[dimension_index])

            kernel_part = numpy.zeros(shape=kernel_width)
            for size_index in xrange(kernel_part.size):
                kernel_part[size_index] = math.exp(-math.pow(size_index - center_index, 2.0) /       \
                                          (2.0 * math.pow(self._steepnesses[dimension_index], 2.0)))

            # normalize kernel part
            kernel_part *= (1. / sum(kernel_part))

            self._separated_kernel_parts.append(kernel_part)

    def check_dimension_index(self, dimension_index):
        dimensionality = self._kernel.get_dimensionality()
        if not (dimension_index >= 0 and dimension_index < dimensionality):
            print("Error. Kernel mode only supports ", dimensionality, "dimensions.")

class Kernel:
    "n-dimensional kernel"
    
    def __init__(self, dimensionality):
        self._dimensionality = None
        self._dimension_sizes = []
    
    def get_dimension_size(self, dimension_index):
        return self._dimension_sizes[dimension_index]

    def get_dimension_sizes(self):
        return self._dimension_sizes

    def get_dimensionality(self):
        return self._dimensionality
    
    def get_separated_kernel_parts(self, dimension_index):
        pass
    
    

class BoxKernel(Kernel):
    "n-dimensional box kernel"
    
    def __init__(self, dimensionality=None):
        Kernel.__init__(self, 1)
        self._dimensionality = 1
        self._amplitude = 5.0
        self._kernel = numpy.ones(shape=(1)) * self._amplitude
    
    def get_separated_kernel_parts(self, dimension_index):
        return self._kernel
        
    def get_amplitude(self):
        return self._amplitude
    
    def set_amplitude(self, amplitude):
        self._amplitude = amplitude        


class GaussKernel(Kernel):
    "n-dimensional Gauss kernel"

    def __init__(self, dimensionality):
        "Constructor"
        
        Kernel.__init__(self, dimensionality)

        self._modes = []
        self._separated_kernel_parts = []
        self._limit = 0.01

    def add_mode(self, amplitude, steepnesses, shift):
        mode = KernelMode(amplitude, steepnesses, shift, self)
        self._modes.append(mode)

    def calculate(self):
        del self._dimension_sizes[:]
        del self._separated_kernel_parts[:]

        for dimension_index in xrange(self._dimensionality):
            dimension_size = self.compute_dimension_size(dimension_index)
            self._dimension_sizes.append(dimension_size)
            self._separated_kernel_parts.append(numpy.zeros(shape=dimension_size))

        for mode in self._modes:
            mode.calculate_separated_kernel_parts()

            for dimension_index in xrange(self._dimensionality):
                self._separated_kernel_parts[dimension_index] += mode.get_separated_kernel_part(dimension_index)

#                kernel_mode_buffer = kernel_mode_buffer * 
                
#                kernel = mode.get_separated_kernel_part(


#    // assemble the kernel
#    mKernel = cv::Mat::zeros(mKernelParts.at(0).at(0).rows, mKernelParts.at(1).at(0).rows, CV_32FC1);
#    for (unsigned int mode = 0; mode < _mNumModes; mode++)
#    {
#      mKernelBuff.at(mode) = cv::Mat(mKernelParts.at(0).at(mode).rows, mKernelParts.at(1).at(mode).rows, CV_32FC1);
#      mKernelBuff.at(mode) = mKernelParts.at(0).at(mode) * mKernelParts.at(1).at(mode).t();
#      mKernel += mKernelBuff.at(mode);
#      // prepare transposed of second dim.
#      mKernelPartsTransposed.at(mode) = mKernelParts.at(1).at(mode).t();
#    }
        
    def compute_dimension_size(self, dimension_index):
        if not (dimension_index >= 0 and dimension_index < self._dimensionality):
            print("Error. Kernel only supports ", self._dimensionality, " dimensions.")
        
        max_width = 0
        for mode in self._modes:
            steepness = mode.get_steepness(dimension_index)
            amplitude = mode.get_amplitude()

            width = 1
            if (steepness < 10000 and steepness > 0):
                width = round(math.sqrt(2.0 * math.pow(steepness, 2.0) * math.log(math.fabs(amplitude) / self._limit))) + 1

            max_width = int(max(width, max_width))

        return max_width

    def get_separated_kernel_parts(self, dimension_index):
        return self._separated_kernel_parts[dimension_index]

