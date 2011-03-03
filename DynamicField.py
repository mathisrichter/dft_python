import math
import random
import GaussKernel
from scipy import ndimage
import numpy

def sigmoid(x, beta, x0):
    return 1./ (1. + numpy.exp(-beta * (x - x0)))


def connect(source, target):
    source_output_dimensionality = source.get_output_dimensionality()
    target_input_dimensionality = target.get_input_dimensionality()

    if source_output_dimensionality != None and target_input_dimensionality != None:
        if source_output_dimensionality != target_input_dimensionality:
            print 'Error. Source and target cannot be connected due to mismatching dimensionality of output and input.'


    source_output_dimension_sizes = source.get_output_dimension_sizes()
    target_input_dimension_sizes = target.get_input_dimensionality()

    if source_output_dimension_sizes != None and target_input_dimension_sizes != None:
        for output_size, input_size in source_output_dimension_sizes, target_intput_dimension_sizes:
            if output_size != input_size:
                print 'Error. Source and target cannot be connected due to mismatching size of at least one dimension.'

    source.get_adjacent_connectables().append(target)
    target.get_incident_connectables().append(source)

def disconnect(source, target):
    source.get_adjacent_connectables().remove(target)
    target.get_incident_connectables().remove(source)



class Connectable:
    "Object that can be connected to other connectable objects via connections."

    def __init__(self):
        # list of connected objects that produce input for this connectable (incident)
        self._incident_connectables = []
        # list of connected objects that receive input from this connectable (adjacent)
        self._adjacent_connectables = []
        # the buffer for the output
        self._output_buffer = None

    def get_incident_connectables(self):
        return self._incident_connectables

    def get_adjacent_connectables(self):
        return self._adjacent_connectables

    def get_output(self):
        return self._output_buffer

    def get_input_dimensionality(self):
        pass

    def get_output_dimensionality(self):
        pass

    def get_input_dimension_sizes(self):
        pass

    def get_output_dimension_sizes(self):
        pass


class DynamicField(Connectable):
    "Dynamic field"

    _instance_counter = 0

    def __init__(self, dimension_sizes, interaction_kernel=None):
        "Constructor"
        Connectable.__init__(self)

        # seed the random number generator to have pseudo-random noise
        random.seed()

        # increase the instance counter of the DynamicField class
        DynamicField._instance_counter += 1

        # unique id of the field instance
        self._id = DynamicField._instance_counter
        
        # name of the field
        self._name = str('')

        # dimensionality of the field
        self._dimensionality = len(dimension_sizes)

        # sizes in each dimension
        self._dimension_sizes = dimension_sizes
         
        # amount of self excitation of the system
        self._lateral_interaction = numpy.zeros(shape=dimension_sizes)
        # convolution kernel used to generate lateral interaction
        self._lateral_interaction_kernel = interaction_kernel

        # noise strength of the system
        self._noise_strength = 0.05
        # value the activation will relax to without external input
        self._resting_level = -5.0
        # input that comes from outside the system and can be used to drive the
        # field through the detection instability
        self._boost = 0.0
        # current value of the system (initialize it with the resting level,
        # because the field would relax to it without external input anyway)
        self._activation = numpy.zeros(shape=dimension_sizes) + self._resting_level

        # controls how fast the system relaxes
        self._relaxation_time = 20.0
        # controls the steepness of the sigmoid (nonlinearity) at the zero
        # crossing
        self._sigmoid_steepness = 5.0
        # controls the shift of the nonlinearity on the x-axis (sigmoid)
        self._sigmoid_shift = 0.0

        # file handle for the activation log
        self._activation_log_file = None

    def __del__(self):
        self.stop_activation_log()

    def get_name(self):
        return self._name

    def set_name(self, name):
        self._name = name

    def get_dimensionality(self):
        return self._dimensionality

    def get_output_dimensionality(self):
        return self.get_dimensionality()

    def get_input_dimensionality(self):
        return self.get_dimensionality()

    def get_dimension_sizes(self):
        return self._dimension_sizes

    def get_input_dimension_sizes(self):
        return self.get_dimension_sizes()

    def get_output_dimension_sizes(self):
        return self.get_dimension_sizes()

    def get_id(self):
        return self._id

    def get_lateral_interaction(self):
        return self._lateral_interaction

    def set_lateral_interaction(self, lateral_interaction):
        self._lateral_interaction = lateral_interaction

    def get_lateral_interaction_kernel(self):
        return self._lateral_interaction

    def set_lateral_interaction_kernel(self, lateral_interaction_kernel):
        self._lateral_interaction_kernel = lateral_interaction_kernel

    def get_noise_strength(self):
        return self._noise_strength

    def set_noise_strength(self, noise_strength):
        self._noise_strength = noise_strength

    def get_resting_level(self):
        return self._resting_level

    def set_resting_level(self, resting_level):
        self._resting_level = resting_level

    def get_boost(self):
        return self._boost

    def set_boost(self, boost):
        self._boost = boost

    def get_activation(self):
        return self._activation

    def get_relaxation_time(self):
        return self._relaxation_time

    def set_relaxation_time(self, relaxation_time):
        self._relaxation_time = relaxation_time

    def get_sigmoid_steepness(self):
        return self._sigmoid_steepness

    def set_sigmoid_steepness(self, steepness):
        self._sigmoid_steepness = steepness

    def get_sigmoid_shift(self):
        return self._sigmoid_shift

    def set_sigmoid_shift(self, shift):
        self._sigmoid_shift = shift

    def get_output(self, activation=None):
        """Compute the output of the field. By default, the current value of the
        field is used, but a different value can be supplied to compute the
        output for an arbitrary value."""
        # if the current value is not supplied..
        if activation is None:
            # ..set it to the current activation of the field
            activation = self._activation

        return sigmoid(activation, self._sigmoid_steepness, self._sigmoid_shift)

    def get_change(self, activation=None, use_time_scale=True):
        """Compute the next change to the system. By default, the current value
        of the field is used, but a different value can be supplied to compute
        the output for an arbitrary value. When use_time_scale is set to False,
        the time scale is not considered when computing the change."""
        # if the current value is not supplied..
        if activation is None:
            # ..set it to the current value of the field
            activation = self._activation

        # if the time scale is to be used..
        if use_time_scale is True:
            # ..compute the inverse of the time scale to have a factor..
            relaxation_time_factor = 1. / self._relaxation_time
        else:
            # ..otherwise, set the factor to one
            relaxation_time_factor = 1.

        # compute the lateral interaction
        if self._lateral_interaction_kernel is not None:
            self._lateral_interaction = GaussKernel.convolve(self.get_output(activation), self._lateral_interaction_kernel)

        # sum up the input coming in from all connected fields
        field_interaction = 0
        for connectable in self.get_incident_connectables():
            field_interaction += connectable.get_output()

        # compute the change of the system
        change = relaxation_time_factor * (- activation
                     + self._resting_level
                     + self._boost
                     + self._lateral_interaction
                     + field_interaction)

        return change
    
    def step(self):
        """Compute the current change of the system and change to current value
        accordingly."""
        self._activation += self.get_change(self._activation) + self._noise_strength * (random.random() - 0.5)
        self.write_activation_log()

    def start_activation_log(self, file_name):
        """Opens the file with the supplied file name. Once there is an open
        file handle, the step() method will write the current activation of the
        field to the file."""
        self._activation_log_file = open(file_name, 'a')

    def stop_activation_log(self):
        "Closes the file handle of the activation log (if it's open)."
        if self._activation_log_file != None:
            self._activation_log_file.close()

    def write_activation_log(self):
        "Writes the current activation of the field to file."
        if self._activation_log_file != None:
            self._activation.tofile(self._activation_log_file, sep=', ')
            self._activation_log_file.write('\n')


class ProcessingGroup(Connectable):
    "A group of processing steps"

    def __init__(self):
        Connectable.__init__(self)
        self._processing_steps = []

    def add_processing_step(self, processing_step, position = None):
        if position is None:
            position = len(self._processing_steps)
        self._processing_steps.insert(position, processing_step)

    def connect_group(self):
        number_of_steps = len(self._processing_steps)

        if number_of_steps > 0:
            connect(self, self._processing_steps[0])
            connect(self._processing_steps[-1], self)

        for step_index in xrange(number_of_steps):
            if step_index + 1 < number_of_steps:
                connect(self._processing_steps[step_index], self._processing_steps[step_index + 1])

    
    def disconnect_group(self):
        number_of_steps = len(self._processing_steps)
        for step_index in xrange(number_of_steps):
            if step_index + 1 < number_of_steps:
                disconnect(self._processing_steps[step_index], self._processing_steps[step_index + 1])

    def step(self):
        for processing_step in self._processing_steps:
            processing_step.step()


class ProcessingStep(Connectable):
    "Processing step"

    def __init__(self):
        Connectable.__init__(self)

    def step(self):
        pass


class WeightProcessingStep(ProcessingStep):
    "The (single) input is multiplied with a weight."

    def __init__(self, weight):
        ProcessingStep.__init__(self)
        self._weight = weight

    def step(self):
        self._output_buffer = self._incident_connectables[0].get_output() * self._weight
