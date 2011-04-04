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

        self._field_dimensionality = field_dimensionality
        self._field_sizes = field_sizes
        self._field_resolutions = field_resolutions
        self._int_node_to_int_field_weight = DynamicField.Weight(int_node_to_int_field_weight)
        self._cos_field_to_cos_node_weight = DynamicField.Weight(cos_field_to_cos_node_weight)
        self._cos_node_to_cos_memory_node_weight = DynamicField.Weight(cos_node_to_cos_memory_node_weight)
        self._int_inhibition_weight = DynamicField.Weight(int_inhibition_weight)
        self._reactivating = reactivating

        self._intention_node_kernel = Kernel.BoxKernel()
        self._intention_node = DynamicField.DynamicField([], [], self._intention_node_kernel)
        self._intention_field_kernel = Kernel.GaussKernel(self._field_dimensionality)
        self._intention_field = DynamicField.DynamicField(field_sizes, field_resolutions, self._intention_field_kernel)

        self._cos_node_kernel = Kernel.BoxKernel()
        self._cos_node = DynamicField.DynamicField([], [], self._cos_node_kernel)
        self._cos_field_kernel = Kernel.GaussKernel(self._field_dimensionality)
        self._cos_field = DynamicField.DynamicField(field_sizes, field_resolutions, self._cos_field_kernel)

        self._cos_memory_node_kernel = Kernel.BoxKernel()
        self._cos_memory_node = DynamicField.DynamicField([], [], self._cos_memory_node_kernel)

        self._connect()

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
           
            

        

