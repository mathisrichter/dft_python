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


class DynamicField:
    "Dynamic field"

    def __init__(self, dimension_sizes, unique_id):
        "Constructor"

        # seed the random number generator to have pseudo-random noise
        random.seed()
        
        # name of the field
        self._name = str('')

        # dimensionality of the field
        # TODO check that dimensionality is correct ( 0 <= dim <= 3 )
        self._dimensionality = dimensionality

        # unique id of the field
        self._id = unique_id
         
        # amount of self excitation of the system
        self._lateral_interaction = numpy.zeros(dimension_sizes) # TODO: convert to tensor
        for 


        # noise strength of the system
        self._noise_strength = 0.05
        # value the activation will relax to without external input
        self._resting_level = -5.0
        # input that comes from outside the system and can be used to drive the
        # field through the detection instability
        self._boost = 0.0
        # current value of the system (initialize it with the resting level,
        # because the field would relax to it without external input anyway)
        self._activation = numpy.zeros(dimension_sizes) + self._resting_level
        # controls how fast the system relaxes
        self._relaxation_time = 20.0
        # controls the steepness of the sigmoid (nonlinearity) at the zero
        # crossing
        self._sigmoid_steepness = 5.0
        # controls the shift of the nonlinearity on the x-axis (sigmoid)
        self._sigmoid_shift = 0.0
        # list of connectors that come into the field
        self._incident_connectors = []
        # list of connectors that go out from the field
        self._adjacent_connectors = []


    def get_name(self):
        return self._name

    def set_name(self, name):
        self._name = name

    def get_dimensionality(self):
        return self._dimensionality

    def get_dimensionality_sizes(self):
        return self._dimensionality_sizes

    def get_id(self):
        return self._id

    def get_lateral_interaction(self):
        return self._lateral_interaction

    def set_lateral_interaction(self, lateral_interaction):
        self._lateral_interaction = lateral_interaction

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

    def get_incident_connectors(self):
        return self._incident_connectors

    def get_adjacent_connectors(self):
        return self._adjacent_connectors

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

        # sum up the input coming in from all connected fields
        field_interaction = 0
        for connector in self._incident_connectors:
            field_interaction += connector.get_output()

        # compute the change of the system
        change = time_scale_factor * (- activation
                     + self._resting_level
                     + self._boost
                     + numpy.convolve(self._lateral_interaction, self.get_output(activation), mode='same')
                     + field_interaction)

        return change
    
    def step(self):
        """Compute the current change of the system and change to current value
        accordingly."""
        self._activation += self.get_change(self._activation) + self._noise * (random.random() - 0.5)


class Connector:
    "Directed connection between two dynamical fields"

    def __init__(self, source_field, target_field, weight, unique_id):
        "Constructor"

        # weight of the connection
        self._weight = weight

        # field the connection originates at
        self._source_field = source_field
        # field the connection ends at
        self._target_field = target_field

        # add the connection to the outgoing connections of the source field
        self._source_field.get_adjacent_connections().append(self)
        # add the connection to the incoming connections of the target field
        self._target_field.get_incident_connections().append(self)

        # unique id of the connection
        self._id = unique_id

    def get_source_field(self):
        return self._source_field

    def get_target_field(self):
        return self._target_field

    def get_weight(self):
        return self._weight

    def set_weight(self, weight):
        self._weight = weight
