import math
import random
import Kernel
from scipy import ndimage
import numpy
import copy

def sigmoid(x, beta, x0):
    return 1./ (1. + numpy.exp(-beta * (x - x0)))

class ConnectError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def connect(source, target, processing_steps):
    # get dimensionality and dimension sizes of source
    source_output_dimensionality = source.get_output_dimensionality()
    source_output_dimension_sizes = source.get_output_dimension_sizes()
    
    # get dimensionality and dimension sizes of target
    target_input_dimensionality = target.get_input_dimensionality()
    target_input_dimension_sizes = target.get_input_dimension_sizes()

    # check if the dimensionality and dimension sizes of the source and the target are known
    if source_output_dimensionality is None:
        raise ConnectError("Source dimensionality is unknown. Cannot set up architecture without this information.")
    if source_output_dimension_sizes is None:
        raise ConnectError("Source dimension sizes are unknown. Cannot set up architecture without this information.")
    if target_input_dimensionality is None:
        raise ConnectError("Target dimensionality is unknown. Cannot set up architecture without this information.")
    if target_input_dimension_sizes is None:
        raise ConnectError("Target dimension sizes are unknown. Cannot set up architecture without this information.")

    # get the indeces of scaler and projection processing steps
    projection_indeces = []
    expanding_projection_indeces = []
    scaler_indeces = []
    for i in xrange(len(processing_steps)):
        if (isinstance(processing_steps[i], Scaling)):
            scaler_indeces.append(i)
        if (isinstance(processing_steps[i], Projection)):
            projection_indeces.append(i)
            if (processing_steps[i].projection_expands() is True):
               expanding_projection_indeces.append(i) 
    
    # print error if there is more than one scaler in the processing steps
    if (len(scaler_indeces) > 1):
        raise ConnectError("You want to connect more than one scaler between source and target. This is not supported.")

    # check that there is no scaler or other projection after the last expanding projection
    if (len(expanding_projection_indeces) > 0):
        if ((len(scaler_indeces) > 0 and expanding_projection_indeces[-1] < scaler_indeces[-1]) or
            expanding_projection_indeces[-1] < projection_indeces[-1]):
            raise ConnectError("""You are trying to put a scaler processing step or
                               a an additional projection after an expanding
                               projection. This is not supported.""")
        
    # if the source and target dimensionalities do not match ..
    if (source_output_dimensionality != target_input_dimensionality):
        # .. check that there is at least one projection in place.
        if (len(projection_indeces) == 0):
            raise ConnectError("""You need a projection processing step to connect source and target,
                               because they are of different dimensionality.""")
    # if the source and target dimensionalities match ..
    else:
        # .. but the sizes of the dimensions do not match ..
        if (source_output_dimension_sizes != target_input_dimension_sizes):
            # .. check that there is either a projection or a scaler in place.
            if (len(scaler_indeces) == 0 and len(projection_indeces) == 0):
                raise ConnectError("""You need a scaler processing step to connect source and target,
                                   because they differ in the size of at least one dimension.
                                   Alternatively, you might also need a projection, which only transposes
                                   the given input.""")
    
    # create a list containing all connectables that are to be connected (source, all processing steps, and target)
    connectables = copy.copy(processing_steps)
    connectables.insert(0, source)
    connectables.append(target)
    
    for i in xrange(len(connectables)-1):
        # check that the dimensionalities match pairwise in the sequence of connectables
        if (connectables[i].get_output_dimensionality() != connectables[i+1].get_input_dimensionality()):
            raise ConnectError("The dimensionality of the connectables " + connectables[i].get_name()
                               + " and " + connectables[i+1].get_name() + " do not match.")
        
        # connect this connectable and the next
        connectables[i].add_outgoing_connectable(connectables[i+1])
        connectables[i+1].add_incoming_connectable(connectables[i])

    # determine the input and output dimension sizes of all connectables
    for i in xrange(1, len(connectables)-1):
        last_output_dimension_sizes = connectables[i-1].get_output_dimension_sizes()
        current_input_dimension_sizes = connectables[i].get_input_dimension_sizes()
        current_output_dimension_sizes = connectables[i].get_output_dimension_sizes()
        next_input_dimension_sizes = connectables[i+1].get_input_dimension_sizes()

        if (current_input_dimension_sizes == None):
            connectables[i].set_input_dimension_sizes(last_output_dimension_sizes)
        else:
            if (current_input_dimension_sizes != last_output_dimension_sizes):
                raise ConnectError("The sizes of at least one dimension do not match between the connectables "
                                   + connectables[i-1].get_name() + "and " + connectables[i].get_name() + ".")
        
        if (current_output_dimension_sizes == None):
            connectables[i].determine_output_dimension_sizes()            
            current_output_dimension_sizes = connectables[i].get_output_dimension_sizes()


def disconnect(source, target):
    source.get_outgoing_connectables().remove(target)
    target.get_incoming_connectables().remove(source)



class Connectable:
    "Object that can be connected to other connectable objects via connections."

    def __init__(self):
        # list of connected objects that produce input for this connectable (incoming)
        self._incoming_connectables = []
        # list of connected objects that receive input from this connectable (outgoing)
        self._outgoing_connectables = []
        # the buffer for the output
        self._output_buffer = None
        
        # dimensionality of the input
        self._input_dimensionality = None
        # dimensionality of the output
        self._output_dimensionality = None
        # dimension sizes of the input
        self._input_dimension_sizes = None
        # dimension sizes of the output
        self._output_dimension_sizes = None

    def get_incoming_connectables(self):
        return self._incoming_connectables
    
    def add_incoming_connectable(self, source):
        self._incoming_connectables.append(source)

    def get_outgoing_connectables(self):
        return self._outgoing_connectables

    def add_outgoing_connectable(self, target):
        self._outgoing_connectables.append(target)
    
    def set_name(self, name):
        self._name = name
    
    def get_name(self):
        return self._name

    def get_output(self):
        return self._output_buffer

    def get_input_dimensionality(self):
        return self._input_dimensionality

    def get_output_dimensionality(self):
        return self._output_dimensionality

    def get_input_dimension_sizes(self):
        return self._input_dimension_sizes

    def get_output_dimension_sizes(self):
        return self._output_dimension_sizes
    
    def set_input_dimensionality(self, dimensionality):
        self._input_dimensionality = dimensionality

    def set_output_dimensionality(self, dimensionality):
        self._output_dimensionality = dimensionality

    def set_input_dimension_sizes(self, dimension_sizes):
        self._input_dimension_sizes = dimension_sizes

    def set_output_dimension_sizes(self, dimension_sizes):
        self._output_dimension_sizes = dimension_sizes
    
    def determine_output_dimension_sizes(self):
        self._output_dimension_sizes = self._input_dimension_sizes


class DynamicField(Connectable):
    "Dynamic field"

    _instance_counter = 0

    def __init__(self, dimension_sizes=[], interaction_kernel=None):
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
        self._input_dimensionality = len(dimension_sizes)
        self._output_dimensionality = self._input_dimensionality

        # sizes in each dimension (in fields, input and output have the same dimensionality)
        if (self._input_dimensionality == 0):
            dimension_sizes = [1]
        self._input_dimension_sizes = dimension_sizes
        self._output_dimension_sizes = dimension_sizes
         
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
        return self._output_dimensionality

    def get_id(self):
        return self._id

    def get_lateral_interaction(self):
        return self._lateral_interaction

    def set_lateral_interaction(self, lateral_interaction):
        self._lateral_interaction = lateral_interaction

    def get_lateral_interaction_kernel(self):
        return self._lateral_interaction_kernel

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
            self._lateral_interaction = Kernel.convolve(self.get_output(activation), self._lateral_interaction_kernel)

        # sum up the input coming in from all connected fields
        field_interaction = 0
        for connectable in self.get_incoming_connectables():
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


class Weight(ProcessingStep):
    "Each input is multiplied with a weight and stored in the corresponding output buffer."

    def __init__(self, weight):
        ProcessingStep.__init__(self)
        self._weight = weight

    def step(self):
        self._output_buffer = self._incoming_connectables[i].get_output() * self._weight
    
    def get_weight(self):
        return self._weight
    
    def set_weight(self, weight):
        self._weight = weight

class Scaling(ProcessingStep):
    """The input is somehow mapped onto an output with different dimension sizes but the same dimensionality.
    This can be done by interpolation, cropping, or padding."""

    def __init__(self, weight):
        ProcessingStep.__init__(self)

    def step(self):
        pass

class Projection(ProcessingStep):
    "Projection of an input onto an output of a different dimensionality."
    
    def __init__(self, input_dimensionality, output_dimensionality, input_dimensions=set(), output_dimensions=[]):
        ProcessingStep.__init__(self)
        self._input_dimensionality = input_dimensionality
        self._output_dimensionality = output_dimensionality
        self._input_dimensions = input_dimensions
        self._output_dimensions = output_dimensions
                
        if (len(self._input_dimensions) > self._input_dimensionality):
            raise ConnectError("Number of input dimensions is larger than the dimensionality of the input.")
        
        if (len(self._output_dimensions) > self._output_dimensionality):
            raise ConnectError("Number of output dimensions is larger than the dimensionality of the output.")

        if (len(self._input_dimensions) != len(self._output_dimensions)):
            raise ConnectError("Number of input dimensions should always be equal to the number of output dimensions.")

        if (max(self._input_dimensions) >= self._input_dimensionality):
            raise ConnectError("""At least one of the indices of your selected input dimensions is higher than the
                               input dimensionality.""")

        if (max(self._output_dimensions) >= self._output_dimensionality):
            raise ConnectError("""At least one of the indices of your selected output dimensions is higher than the
                               output dimensionality.""")
        
        if (self._input_dimensionality == self._output_dimensionality):
            if (self._input_dimensions == self._output_dimensions):
                print """Warning. You have created a projection processing step that neither changes the dimensionality,
                       nor reorders the dimension indices and thus does nothing but waste processing power. :)"""

        self._projection_compresses = False
        self._projection_expands = False

        if (len(self._input_dimensions) < self._input_dimensionality):
            self._projection_compresses = True
            
        if (len(self._output_dimensions) < self._output_dimensionality):
            self._projection_expands = True

        if (self._projection_compresses and self._projection_expands):
            raise ConnectError("""The projection is set up to both compress the input and expand it afterwards.
                               This is not supported. Please use two separate projection processing steps to
                               achieve the same effect.""")

        # get a set (unordered) of dimensions, which will be compressed (i.e., projected onto the remaining dimensions).
        # if no input dimensions are given, then the whole input will be compressed to a scalar.
        self._dimensions_to_compress = None
        if (self._projection_compresses):
            self._dimensions_to_compress = list(set(range(input_dimensionality)).difference(set(input_dimensions)))

        self._expand_method = None
        self._transpose_permutation = None
        self._inverse_transpose_permutation = None
        if (self._projection_expands):
            if (self._input_dimensionality == 0):
                self._expand_method = self._expand_0D
            elif (self._input_dimensionality == 1):
                if (self._output_dimensionality == 2):
                    self._expand_method = self._expand_1D_2D
                elif (self._output_dimensionality == 3):
                    self._expand_method = self._expand_1D_3D

                    indeces = range(self._output_dimensionality)
                    for d in self._output_dimensions:
                        indeces.remove(d)

                    third_dimension_index = indeces[0]
                    second_dimension_index = indeces[1]
                    first_dimension_index = self._output_dimensions[0]
                    self._transpose_permutation = (third_dimension_index, second_dimension_index, first_dimension_index)
                    self._inverse_transpose_permutation = self._invert_permutation(self._transpose_permutation)
                else:
                    raise ConnectError("""You are trying to expand a 1D input to
                                       something other than 2D or 3D. This is not yet supported.""")
            elif (self._input_dimensionality == 2):
                if (self._output_dimensionality == 3):
                    self._expand_method = self._expand_2D_3D

                    indeces = range(self._output_dimensionality)
                    for d in self._output_dimensions:
                        indeces.remove(d)

                    third_dimension_index = indeces[0]
                    self._transpose_permutation = copy.copy(self._output_dimensions)
                    self._transpose_permutation.insert(0, third_dimension_index)
                    self._inverse_transpose_permutation = self._invert_permutation(self._transpose_permutation)
                else:
                    raise ConnectError("""You are trying to expand a 2D input to
                                       something other than 3D. This is not yet supported.""")
        

    def step(self):
        input = self._incoming_connectables[0].get_output()

        if (self._projection_compresses):
            for i in xrange(len(self._dimensions_to_compress)):
                input = input.max(self._dimensions_to_compress[i] - i)
        elif (self._projection_expands):
            self._output_buffer = self._expand_method(input)

        if (self._projection_expands is not True):
            self._output_buffer = numpy.transpose(input, self._output_dimensions)
    
    def determine_output_dimension_sizes(self):
        if (len(self._outgoing_connectables) > 0):
            if (self._projection_expands):
                next_input_dimension_sizes = self._outgoing_connectables[0].get_input_dimension_sizes()
                if (next_input_dimension_sizes is None):
                    raise ConnectError("""The connectable after the expanding
                                       projection does not know its input
                                       dimension sizes. This should not
                                       happen.""")
                self._output_dimension_sizes = next_input_dimension_sizes
            else:
                selected_dimension_sizes = []
                
                for input_dimension in self._input_dimensions:
                    selected_dimension_sizes.append(self._input_dimension_sizes[input_dimension])
                
                self._output_dimension_sizes = [selected_dimension_sizes[i] for i in self._output_dimensions]
        else:
            raise ConnectError("""You are trying to determine the output
                               dimension size of a connectable that does not
                               have an outgoing connection to another
                               connectable. This should not happen.""")

    def determine_input_dimension_sizes(self):
        pass

    def projection_expands(self):
        return self._projection_expands

    def projection_compresses(self):
        return self._projection_compresses

    def _expand_0D(self, input):
        return numpy.zeros(self._output_dimension_sizes) + input

    def _expand_1D_2D(self, input):
        output = numpy.zeros(self._output_dimension_sizes)

        transpose = False
        if (self._output_dimensions[0] == 0):
            transpose = True
            output = numpy.transpose(output)

        for i in xrange(len(output)):
            output[i] = input

        if (transpose):
            output = numpy.transpose(output)

        return output

    def _expand_1D_3D(self, input):
        output = numpy.zeros(self._output_dimension_sizes)

        if (self._transpose_permutation is not None):
            output = numpy.transpose(output, self._transpose_permutation)

        second_dimension_size = self._output_dimension_sizes[self._transpose_permutation[1]]
        first_dimension_size = self._output_dimension_sizes[self._transpose_permutation[2]]

        two_dim_activation = numpy.zeros((second_dimension_size, first_dimension_size))
        for i in xrange(len(two_dim_activation)):
            two_dim_activation[i] = input

        for i in xrange(len(output)):
            output[i] = two_dim_activation

        if (self._transpose_permutation is not None):
            output = numpy.transpose(output, self._inverse_transpose_permutation)

        return output

    def _expand_2D_3D(self, input):
        output = numpy.zeros(self._output_dimension_sizes)

        if (self._transpose_permutation is not None):
            output = numpy.transpose(output, self._transpose_permutation)

        for i in xrange(len(output)):
            output[i] = input

        if (self._transpose_permutation is not None):
            output = numpy.transpose(output, self._inverse_transpose_permutation)

        return output

    def _invert_permutation(self, permutation):
        inverse_permutation = []
        for i in range(len(permutation)):
            inverse_permutation.append(permutation.index(i))

        return inverse_permutation

