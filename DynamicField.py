import math
import random
import GaussKernel
from scipy import ndimage
import numpy

def sigmoid(x, beta, x0):
    return 1./ (1. + numpy.exp(-beta * (x - x0)))


class DynamicField:
    "Dynamic field"

    _instance_counter = 0

    def __init__(self, dimension_sizes, interaction_kernel=None):
        "Constructor"

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
        # list of connectors that come into the field (incident connectors)
        self._incoming_connectors = []
        # list of connectors that go out from the field (adjacent connectors)
        self._outgoing_connectors = []


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

    def get_incoming_connectors(self):
        return self._incoming_connectors

    def get_outgoing_connectors(self):
        return self._outgoing_connectors

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
        field_interaction = 10
        for connector in self._incoming_connectors:
            field_interaction += connector.get_output()

        # compute the change of the system
        change = relaxation_time_factor * (- activation
                     + self._resting_level
                     + self._boost
                     + self._lateral_interaction
                     + field_interaction)

        print('change: ', change)

        return change
    
    def step(self):
        """Compute the current change of the system and change to current value
        accordingly."""
        self._activation += self.get_change(self._activation) + self._noise_strength * (random.random() - 0.5)


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
