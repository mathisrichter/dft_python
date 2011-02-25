import math
import random
from scipy import ndimage
from scipy import numpy

def sigmoid(x, beta, x0):
    return 1./ (1. + math.exp(-beta * (x - x0)))

def convolve(input, kernel):
    if kernel.is_separable() is True
        convolution_result = copy(input)
        for dimension_index in range(0, kernel.get_number_of_dimensions):
            ndimage.convolve1d(convolution_result,
                kernel.get_separated_kernel_parts(dimension_index),
                axis=dimension_index,
                output=convolution_result,
                mode='wrap')
    else
        print "Error. For now, the convolution is only implemented for separable filters."

    return convolution_result

class KernelMode:
    "Mode of a kernel"

    def __init__(self, amplitude, steepnesses, shifts):
        self._amplitude = amplitude 
        self._steepnesses = steepnesses
        self._shifts = shifts
        self._number_of_dimensions = len(steepnesses)
        if len(shifts) != self._number_of_dimensions
            print "Error. Kernel mode only supports ", len(self._steepnesses), "dimensions."
            
        self._separated_kernel_parts = []
        self._is_separable = False

    def is_separable(self):
        return self._is_separable

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
        self._is_separable = True

        for dimension_index in range(0, number_of_dimensions):
            kernel_width = compute_kernel_width(dimension_index)
            self._dimension_sizes[dimension_index] = kernel_width

            center_index = kernel_width / 2 + mode.get_shift(dimension_index) 
            mode.set_center_index(center_index, dimension_index)

            kernel_part = []
            for size_index in range(0, kernel_width)
                value = exp(-math.pow(size_index - center_index, 2.0) / (2.0 * math.pow(mode.get_steepness(dimension_index), 2.0)))
                kernel_part.append(value)

            # normalize kernel part
            kernel_part *= (1. / sum(kernel_part))

            self._separated_kernel_part[dimension_index] = kernel_part

       
    def check_dimension_index(self, dimension_index):
        if not (dimension_index >= 0 and dimension_index < len(self._steepnesses)):
            print "Error. Kernel mode only supports ", len(self._steepnesses), "dimensions."



class GaussKernel:
    "n-dimensional Gauss kernel"

    def __init__(self, number_of_dimensions):
        "Constructor"

        self._number_of_dimensions = number_of_dimensions
        self._dimension_sizes = []
        self._modes = []
        self._limit = 0.01

    def add_mode(self, amplitude, steepnesses, shift):
        mode = KernelMode(amplitude, steepnesses, shift)
        self.add_mode(mode)

    def calculate(self):
        for mode in self._modes:

        for mode in self._modes:
            kernel_mode_buffer = numpy.eye(shape=self._dimension_sizes)

            for dimension_index in range(0, number_of_dimensions):
                kernel_mode_buffer = kernel_mode_buffer * 
                
                kernel = mode.get_separated_kernel_part(


    // assemble the kernel
    mKernel = cv::Mat::zeros(mKernelParts.at(0).at(0).rows, mKernelParts.at(1).at(0).rows, CV_32FC1);
    for (unsigned int mode = 0; mode < _mNumModes; mode++)
    {
      mKernelBuff.at(mode) = cv::Mat(mKernelParts.at(0).at(mode).rows, mKernelParts.at(1).at(mode).rows, CV_32FC1);
      mKernelBuff.at(mode) = mKernelParts.at(0).at(mode) * mKernelParts.at(1).at(mode).t();
      mKernel += mKernelBuff.at(mode);
      // prepare transposed of second dim.
      mKernelPartsTransposed.at(mode) = mKernelParts.at(1).at(mode).t();
    }
        


    def compute_kernel_width(self, dimension_index):
        
        if not (dimension_index >= 0 and dimension_index < number_of_dimensions):
            print "Error. Kernel only supports ", number_of_dimensions, " dimensions."
        
        max_width = 0
        for mode in self._modes:
            steepness = mode.get_steepness(dimension_index)
            amplitude = mode.get_amplitude()

            width = 1
            if (steepness < 10000 and steepness > 0):
                width = round(math.sqrt(2.0 * pow(steepness, 2.0) * math.log(math.abs(amplitude) / self._limit))) + 1

            max_width = max(width, max_width)

        return max_width

    

