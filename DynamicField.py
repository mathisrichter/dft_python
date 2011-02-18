import math
import random

def sigmoid(x, beta, x0):
    return 1./ (1. + math.exp(-beta * (x - x0)))

class Node:
    "Dynamical field"

    def __init__(self, unique_id):
        "Constructor"

        # seed the random number generator to have pseudo-random noise
        random.seed()
        
        # name of the field
        self._name = str('')

        # unique id of the field
        self._id = unique_id
        
        # amount of self excitation of the system
        self._self_excitation = 5.0
        # noise of the system
        self._noise = 0.05
        # value the activation will relax to without external input
        self._resting_level = -5.0
        # input that comes from outside the system
        self._external_input = 0.0
        # current value of the system (initialize it with the resting level,
        # because the field would relax to it without external input anyway)
        self._current_value = self._resting_level
        # controls how fast the system relaxes
        self._time_scale = 20.0
        # controls the steepness of the sigmoid (nonlinearity) at the zero
        # crossing
        self._sigmoid_steepness = 5.0
        # controls the shift of the nonlinearity on the x-axis (sigmoid)
        self._sigmoid_shift = 0.0
        # list of connections that come into the field
        self._incident_connections = []
        # list connections that go out from the field
        self._adjacent_connections = []

    def get_name(self):
        return self._name

    def set_name(self, name):
        self._name = name

    def get_id(self):
        return self._id

    def get_self_excitation(self):
        return self._self_excitation

    def set_self_excitation(self, self_excitation):
        self._self_excitation = self_excitation

    def get_noise(self):
        return self._noise

    def set_noise(self, noise):
        self._noise = noise

    def get_resting_level(self):
        return self._resting_level

    def set_resting_level(self, resting_level):
        self._resting_level = resting_level

    def get_external_input(self):
        return self._external_input

    def set_external_input(self, external_input):
        self._external_input = external_input

    def get_current_value(self):
        return self._current_value

    def get_time_scale(self):
        return self._time_scale

    def set_time_scale(self, time_scale):
        self._time_scale = time_scale

    def get_sigmoid_steepness(self):
        return self._sigmoid_steepness

    def set_sigmoid_steepness(self, steepness):
        self._sigmoid_steepness = steepness

    def get_sigmoid_shift(self):
        return self._sigmoid_shift

    def set_sigmoid_shift(self, shift):
        self._sigmoid_shift = shift

    def get_incident_connections(self):
        return self._incident_connections

    def get_adjacent_connections(self):
        return self._adjacent_connections

    def get_output(self, current_value=None):
        """Compute the output of the field. By default, the current value of the
        field is used, but a different value can be supplied to compute the
        output for an arbitrary value."""
        # if the current value is not supplied..
        if current_value is None:
            # ..set it to the current value of the field
            current_value = self._current_value

        return sigmoid(current_value, self._sigmoid_steepness, self._sigmoid_shift)

    def get_change(self, current_value=None, use_time_scale=True):
        """Compute the next change to the system. By default, the current value
        of the field is used, but a different value can be supplied to compute
        the output for an arbitrary value. When use_time_scale is set to False,
        the time scale is not considered when computing the change."""
        # if the current value is not supplied..
        if current_value is None:
            # ..set it to the current value of the field
            current_value = self._current_value

        # if the time scale is to be used..
        if use_time_scale is True:
            # ..compute the inverse of the time scale to have a factor..
            time_scale_factor = 1. / self._time_scale
        else:
            # ..otherwise, set the factor to one
            time_scale_factor = 1.

        # sum up the input coming in from all connected fields
        field_interaction = 0
        for connection in self._incident_connections:
            field_interaction += connection.get_weight() * connection.get_source_field().get_output()
        # compute the change of the system
        change = time_scale_factor * (- current_value
                     + self._resting_level
                     + self._external_input
                     + self._self_excitation * self.get_output(current_value)
                     + field_interaction)

        return change
    
    def step(self):
        """Compute the current change of the system and change to current value
        accordingly."""
        self._current_value += self.get_change(self._current_value) + self._noise * (random.random() - 0.5)


class Connection:
    "Weighted, directed connection between two dynamical fields"

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
