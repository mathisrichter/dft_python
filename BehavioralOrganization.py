import math
import random
import Kernel
import numpy
import copy
from DynamicField import DynamicField
from DynamicField import connect

class ElementaryBehavior:
    def __init__(self, field_dimensionality, field_sizes, field_resolutions, reactivating=False):
        self._field_dimensionality = field_dimensionality

        self._intention_node_kernel = Kernel.BoxKernel()
        self._intention_node = DynamicField([], [], self._intention_node_kernel)
        self._intention_field_kernel = Kernel.GaussKernel(self._field_dimensionality)
        self._intention_field = DynamicField(field_sizes, field_resolutions, self._intention_field_kernel)

        self._cos_node_kernel = Kernel.BoxKernel()
        self._cos_node = DynamicField([], [], None)
        self._cos_field_kernel = Kernel.GaussKernel(self._field_dimensionality)
        self._cos_field = DynamicField(field_sizes, field_resolutions, self._cos_field_kernel)

        self._cos_memory_node_kernel = Kernel.BoxKernel()
        self._cos_memory_node = DynamicField([], [], self._cos_memory_node_kernel)

        self._connect()

    def _connect(self):
        # connect intention node to intention field
        intention_weight = DynamicField.Weight(self._intention_weight)
        intention_projection = DynamicField.Projection(0, self._field_dimensionality, set([]), range(self._field_dimensionality))
        intention_processing_steps = [intention_projection, intention_weight]
        DynamicField.connect(self._intention_node, self._intention_field, intention_processing_steps)

        # connect intention field to cos field
        DynamicField.connect(self._intention_field, self._cos_field)

        # connect cos field to cos node
        cos_projection = DynamicField.Projection(self._field_dimensionality, 0, set(range(self._field_dimensionality)), [])
        DynamicField.connect(self._cos_field, self._cos_node, cos_processing_steps)

        # connect cos node to cos memory node
        cos_memory_weight = DynamicField.Weight(self._cos_to_cos_memory_weight)
        DynamicField.connect(self._cos_node, self._cos_memory_node, [cos_memory_weight])

        # connect the node that inhibits the intention node (either cos or cos-memory) with the intention node
        intention_inhibiting_node = self._cos_memory_node
        if (self._reactivating):
            intention_inhibiting_node = self._cos_node

        DynamicField.connect(intention_inhibiting_node, self._intention_node, [self._intention_inhibiting_weight])
            

        

