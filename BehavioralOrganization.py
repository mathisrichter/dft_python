import math
import random
import Kernel
import numpy
import copy
import DynamicField

class ElementaryBehavior:
    def __init__(self,
                 field_dimensionality,
                 field_sizes,
                 field_resolutions,
                 int_node_to_int_field_weight,
                 cos_field_to_cos_node_weight,
                 cos_node_to_cos_memory_node_weight,
                 int_inhibition_weight,
                 reactivating=False):

        # dimensionality of the intention and CoS field (for now, they have the same dimensionality)
        self._field_dimensionality = field_dimensionality
        # sizes of the intention and CoS field (for now, they have the same sizes)
        self._field_sizes = field_sizes
        # resolutions of each dimension of the intention and CoS field (for now, they have the same resolutions)
        self._field_resolutions = field_resolutions
        # does the node reactivate its intention, when the CoS node gets deactivated?
        self._reactivating = reactivating

        # connectables that describe weights between different nodes/fields
        self._int_node_to_int_field_weight = DynamicField.Weight(int_node_to_int_field_weight)
        self._cos_field_to_cos_node_weight = DynamicField.Weight(cos_field_to_cos_node_weight)
        self._cos_node_to_cos_memory_node_weight = DynamicField.Weight(cos_node_to_cos_memory_node_weight)
        self._int_inhibition_weight = DynamicField.Weight(int_inhibition_weight)

        # intention node and its kernel
        self._intention_node_kernel = None
        self._intention_node = DynamicField.DynamicField([], [], self._intention_node_kernel)
        # intention field and its kernel
        self._intention_field_kernel = Kernel.GaussKernel(self._field_dimensionality)
        self._intention_field_kernel.add_mode(1.0, [0.5] * self._field_dimensionality, [0.0] * self._field_dimensionality)
        self._intention_field_kernel.add_mode(-5.5, [5.5] * self._field_dimensionality, [0.0] * self._field_dimensionality)
        self._intention_field_kernel.calculate()
        self._intention_field = DynamicField.DynamicField(field_sizes, field_resolutions, self._intention_field_kernel)
        # CoS node and its kernel
        self._cos_node_kernel = None
        self._cos_node = DynamicField.DynamicField([], [], self._cos_node_kernel)
        # CoS field and its kernel
        self._cos_field_kernel = Kernel.GaussKernel(self._field_dimensionality)
        self._cos_field_kernel.add_mode(1.0, [0.5] * self._field_dimensionality, [0.0] * self._field_dimensionality)
        self._cos_field_kernel.add_mode(-5.5, [5.5] * self._field_dimensionality, [0.0] * self._field_dimensionality)
        self._cos_field_kernel.calculate()
        self._cos_field = DynamicField.DynamicField(field_sizes, field_resolutions, self._cos_field_kernel)
        # CoS memory node and its kernel
        self._cos_memory_node_kernel = None
        self._cos_memory_node = DynamicField.DynamicField([], [], self._cos_memory_node_kernel)

        # connect all connectables in this elementary behavior
        self._connect()

    def get_intention_node(self):
        return self._intention_node

    def get_intention_field(self):
        return self._intention_field

    def get_cos_node(self):
        return self._cos_node

    def get_cos_field(self):
        return self._cos_field

    def get_cos_memory_node(self):
        return self._cos_memory_node

    def _connect(self):
        # connect intention node to intention field
        self._intention_projection = DynamicField.Projection(0, self._field_dimensionality, set([]), [])
        intention_processing_steps = [self._intention_projection, self._int_node_to_int_field_weight]
        DynamicField.connect(self._intention_node, self._intention_field, intention_processing_steps)

        # connect intention field to cos field
        DynamicField.connect(self._intention_field, self._cos_field)

        # connect cos field to cos node
        self._cos_projection = DynamicField.Projection(self._field_dimensionality, 0, set([]), [])
        DynamicField.connect(self._cos_field, self._cos_node, [self._cos_projection, self._cos_field_to_cos_node_weight])

        # connect cos node to cos memory node
        DynamicField.connect(self._cos_node, self._cos_memory_node, [self._cos_node_to_cos_memory_node_weight])

        # connect the node that inhibits the intention node (either cos or cos-memory) with the intention node
        intention_inhibition_node = self._cos_memory_node
        if (self._reactivating):
            intention_inhibition_node = self._cos_node

        DynamicField.connect(intention_inhibition_node, self._intention_node, [self._int_inhibition_weight])

    def step(self):
        connectables = [self._intention_node,
                        self._intention_projection,
                        self._int_node_to_int_field_weight,
                        self._intention_field,
                        self._cos_field,
                        self._cos_projection,
                        self._cos_field_to_cos_node_weight,
                        self._cos_node,
                        self._cos_node_to_cos_memory_node_weight,
                        self._cos_memory_node,
                        self._int_inhibition_weight]
             
        for connectable in connectables:
            connectable.step()
           
            

        

