import math
import random
import numpy

def sigmoid(x, beta, x0):
    return 1./ (1. + math.exp(-beta * (x - x0)))

def gauss(x):
    return 1. / sqrt(2. * PI) * exp(-1./2. * x**2)

def discrete_gauss(x, array_size):
    gauss_array = zeros(array_size)

    for i in range(array_size):
        j = i - array_size / (array_size * 10)
        gauss_array[i] = gauss(j)

    return gauss_array

class KernelMode:
    "Mode of a kernel"

    def __init__(self, amplitude, steepnesses, shift):
        self._amplitude = amplitude 
        self._steepnesses = steepnesses
        self._shift = shift
        self._separated_kernel_parts = []

    def get_amplitude(self):
        return self._amplitude

    def get_steepness(self, dimension_index):
        if not (dimension_index >= 0 and dimension_index < len(self._steepnesses)):
            print "Error. Kernel mode only supports ", len(self._steepnesses), "dimensions."
        return self._steepnesses[dimension_index]

    def get_shift(self):
        return self._shift

    def set_amplitude(self, amplitude):
        self._amplitude = amplitude

    def set_steepness(self, steepness, dimension_index):
        if not (dimension_index >= 0 and dimension_index < len(self._steepnesses)):
            print "Error. Kernel mode only supports ", len(self._steepnesses), "dimensions."
        self._steepnesses[dimension_index] = steepness

    def set_shift(self, shift):
        self._shift = shift


class GaussKernel:
    "n-dimensional Gauss kernel"

    def __init__(self):
        "Constructor"

        self._number_of_dimensions = number_of_dimensions
        self._modes = []
        self._limit = 0.01
        self._dimension_sizes = []

    def add_mode(self, amplitude, steepnesses, shift):
        mode = KernelMode(amplitude, steepnesses, shift)
        self.add_mode(mode)

    def calculate(self):

        for mode in self._modes:
            for dimension_index in range(0, number_of_dimensions):
                kernel_width = compute_kernel_width(dimension_index)
                self._dimension_sizes[dimension_index] = kernel_width

                center_index = kernel_width / 2 + mode.get_shift(dimension_index)
                self.separated_kernel_parts

        mCenters.at(dim).at(mode) = mSizes.at(dim) / 2 + _mShifts.at(dim).at(mode);
        mKernelParts.at(dim).at(mode) = cv::Mat::zeros(mSizes.at(dim), 1, CV_32FC1);
        if (dim == 1)
        {
          // just need transposed parts of second dimension
          mKernelPartsTransposed.at(mode) = cv::Mat(1,mSizes.at(1), CV_32FC1);
        }
        // calculate kernel part
        if (_mSigmas.at(dim).at(mode) != 0)
        {
          for (int j = 0; j < mSizes.at(dim); j++)
          {
            mKernelParts.at(dim).at(mode).at<float>(j, 0)
                = exp(-powf(j - mCenters.at(dim).at(mode), 2) / (2 * powf(_mSigmas.at(dim).at(mode), 2)));
          }
        }
        else // discrete case
        {
          mKernelParts.at(0).at(mode).at<float>(0, 0) = 1;
        }
        // normalize
        mKernelParts.at(dim).at(mode) = mKernelParts.at(dim).at(mode) * (1. / sum(mKernelParts.at(dim).at(mode)).val[0]);
        if(dim == 0)
        {
          mKernelParts.at(dim).at(mode) = _mAmplitudes.at(mode) * mKernelParts.at(dim).at(mode);
        }
      }
    }
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

    

