import math
import random
import Kernel
import numpy
import copy
import DynamicField
import Kernel
import math_tools
import CameraField
import EndEffectorControl


class GraspArchitecture():

    def __init__(self):
        self.fields = []

        # create a task node
        self._task_node = DynamicField.DynamicField([], [], None)
        self._task_node.set_boost(10)

        # create elementary behavior: find color
        self._find_color_field_size = 15
        find_color_int_weight = math_tools.gauss_1d(self._find_color_field_size, amplitude=15.0, sigma=0.5, shift=0)

        self._find_color = ElementaryBehavior.with_internal_fields(field_dimensionality=1,
                                                    field_sizes=[[self._find_color_field_size]],
                                                    field_resolutions=[],
                                                    int_node_to_int_field_weight=find_color_int_weight,
                                                    cos_field_to_cos_node_weight=0,# NEEDS TO BE CHANGED BACK
                                                    name="find color")

        self.fields.append(self._find_color.get_intention_field())
        self.fields.append(self._find_color.get_cos_field())

        # create elementary behavior: move end effector
        self._move_ee_field_sizes = [40, 30]
        move_ee_int_weight = numpy.ones((self._move_ee_field_sizes)) * 4.0
        self._move_ee = ElementaryBehavior.with_internal_fields(field_dimensionality=2,
                                                    field_sizes=[[self._move_ee_field_sizes[0]],[self._move_ee_field_sizes[1]]],
                                                    field_resolutions=[],
                                                    int_node_to_int_field_weight=move_ee_int_weight,
                                                    name="move ee")
        self.fields.append(self._move_ee.get_intention_field())
        self.fields.append(self._move_ee.get_cos_field())

        # create gripper intention and cos fields
        gripper_field_dimensionality = 1
        self._gripper_field_size = 15

        # gripper intention field and its kernel
        intention_field_kernel = Kernel.GaussKernel(gripper_field_dimensionality)
        intention_field_kernel.add_mode(15.0, [1.0] * gripper_field_dimensionality, [0.0] * gripper_field_dimensionality)
        intention_field_kernel.calculate()
        self._gripper_intention_field = DynamicField.DynamicField([[self._gripper_field_size]], [], intention_field_kernel)
        self._gripper_intention_field.set_global_inhibition(200.0)

        # gripper CoS field and its kernel
        cos_field_kernel = Kernel.GaussKernel(gripper_field_dimensionality)
        cos_field_kernel.add_mode(15.0, [1.0] * gripper_field_dimensionality, [0.0] * gripper_field_dimensionality)
        cos_field_kernel.calculate()
        self._gripper_cos_field = DynamicField.DynamicField([[self._gripper_field_size]], [], cos_field_kernel)
        self._gripper_cos_field.set_global_inhibition(200.0)

        # connect the gripper intention and CoS field
        gripper_int_field_to_cos_field_weight = DynamicField.Weight(2.5)
        DynamicField.connect(self._gripper_intention_field, self._gripper_cos_field, [gripper_int_field_to_cos_field_weight])

        # create elementary behavior: gripper close
        gripper_close_int_weight = math_tools.gauss_1d(self._gripper_field_size, amplitude=15, sigma=0.5, shift=5)
        self._gripper_close = ElementaryBehavior(intention_field=self._gripper_intention_field,
                                                  cos_field=self._gripper_cos_field,
                                                  int_node_to_int_field_weight=gripper_close_int_weight,
                                                  name="gripper close")

        # create elementary behavior: gripper open
        gripper_open_int_weight = math_tools.gauss_1d(self._gripper_field_size, amplitude=15, sigma=0.5, shift=self._gripper_field_size-5)
        self._gripper_open = ElementaryBehavior(intention_field=self._gripper_intention_field,
                                                  cos_field=self._gripper_cos_field,
                                                  int_node_to_int_field_weight=gripper_open_int_weight,
                                                  name="gripper open")
        self._gripper_intention_field.set_name("gripper_intention_field")
        self._gripper_cos_field.set_name("gripper_cos_field")

        self.fields.append(self._gripper_intention_field)
        self.fields.append(self._gripper_cos_field)


        # connect all elementary behaviors to the task node
        connect_to_task(self._task_node, self._find_color)
        connect_to_task(self._task_node, self._move_ee)
        connect_to_task(self._task_node, self._gripper_open)
        connect_to_task(self._task_node, self._gripper_close)

        # create precondition nodes
        self._gripper_open_precondition_node = precondition(self._gripper_open, self._move_ee, self._task_node)
        self._gripper_close_precondition_node = precondition(self._move_ee, self._gripper_close, self._task_node)

        # create perception color-space field
        color_space_field_dimensionality = 3
        color_space_kernel = Kernel.GaussKernel(color_space_field_dimensionality)
        color_space_kernel.add_mode(10.0, [1.0] * color_space_field_dimensionality, [0.0] * color_space_field_dimensionality)
        color_space_kernel.calculate()

        self._color_space_field_sizes = [self._move_ee_field_sizes[0], self._move_ee_field_sizes[1], self._find_color_field_size]
        self._color_space_field = DynamicField.DynamicField([[self._color_space_field_sizes[0]],[self._color_space_field_sizes[1]],[self._color_space_field_sizes[2]]], [], color_space_kernel)
        self._color_space_field.set_global_inhibition(800.0)
        self._color_space_field.set_relaxation_time(2.0)
        self._color_space_field.set_name("color_space_field")
        self.fields.append(self._color_space_field)

        fc_int_to_color_space_projection = DynamicField.Projection(self._find_color.get_intention_field().get_dimensionality(), color_space_field_dimensionality, set([0]), [2])
        fc_int_to_color_space_weight = DynamicField.Weight(6.0)
        DynamicField.connect(self._find_color.get_intention_field(), self._color_space_field, [fc_int_to_color_space_weight, fc_int_to_color_space_projection])

        color_space_to_fc_cos_projection = DynamicField.Projection(color_space_field_dimensionality, self._find_color.get_cos_field().get_dimensionality(), set([2]), [0])
        color_space_to_fc_cos_weight = DynamicField.Weight(8.0)
        DynamicField.connect(self._color_space_field, self._find_color.get_cos_field(), [color_space_to_fc_cos_projection, color_space_to_fc_cos_weight])

        # create "camera" field
        self._camera_field = CameraField.NaoCameraField()
        self._camera_field.set_name("camera_field")
        self.fields.append(self._camera_field)
        self._camera_field_sizes = self._camera_field.get_output_dimension_sizes()

        camera_to_color_space_weight = DynamicField.Weight(4.0)
        DynamicField.connect(self._camera_field, self._color_space_field, [camera_to_color_space_weight])

        # create "spatial target location" field
        spatial_target_field_dimensionality = 2
        spatial_target_kernel = Kernel.GaussKernel(spatial_target_field_dimensionality)
        spatial_target_kernel = Kernel.GaussKernel(spatial_target_field_dimensionality)
        spatial_target_kernel.add_mode(2.0, [1.0] * spatial_target_field_dimensionality, [0.0] * spatial_target_field_dimensionality)
        spatial_target_kernel.calculate()

        self._spatial_target_field_sizes = self._move_ee_field_sizes
        self._spatial_target_field = DynamicField.DynamicField([[self._spatial_target_field_sizes[0]], [self._spatial_target_field_sizes[1]]], [], spatial_target_kernel)
        self._spatial_target_field.set_global_inhibition(20.0)
        self._spatial_target_field.set_name("spatial_target_field")
        self.fields.append(self._spatial_target_field)

        color_space_to_spatial_target_projection = DynamicField.Projection(color_space_field_dimensionality, spatial_target_field_dimensionality, set([0, 1]), [0, 1])
        color_space_to_spatial_target_weight = DynamicField.Weight(7.0)
        DynamicField.connect(self._color_space_field, self._spatial_target_field, [color_space_to_spatial_target_projection, color_space_to_spatial_target_weight])

        spatial_target_to_move_ee_int_weight = DynamicField.Weight(7.0)
        DynamicField.connect(self._spatial_target_field, self._move_ee.get_intention_field(), [spatial_target_to_move_ee_int_weight])

        # create perception field in end effector space
        perception_ee_field_dimensionality = 2
        perception_ee_kernel = Kernel.GaussKernel(perception_ee_field_dimensionality)
        perception_ee_kernel = Kernel.GaussKernel(perception_ee_field_dimensionality)
        perception_ee_kernel.add_mode(15.0, [1.0] * perception_ee_field_dimensionality, [0.0] * perception_ee_field_dimensionality)
        perception_ee_kernel.calculate()

        self._perception_ee_field_sizes = self._move_ee_field_sizes
        self._perception_ee_field = DynamicField.DynamicField([[self._perception_ee_field_sizes[0]], [self._perception_ee_field_sizes[1]]], [], perception_ee_kernel)
        self._perception_ee_field.set_global_inhibition(400.0)
        self._perception_ee_field.set_name("perception_ee_field")
        self.fields.append(self._perception_ee_field)

        perception_ee_to_move_ee_cos_weight = DynamicField.Weight(3.5)
        DynamicField.connect(self._perception_ee_field, self._move_ee.get_cos_field(), [perception_ee_to_move_ee_cos_weight])

        # create end effector control connectable
        self._end_effector_control = EndEffectorControl.EndEffectorControl(self._move_ee_field_sizes, head_speed_fraction = 0.4)
        DynamicField.connect(self._move_ee.get_intention_field(), self._end_effector_control)

    def step(self):
        self._task_node.step()
        self._camera_field.step()
        self._find_color.step()
        self._perception_ee_field.step()
        self._color_space_field.step()
        self._spatial_target_field.step()
        self._gripper_intention_field.step()
        self._gripper_cos_field.step()
        self._gripper_open.step()
        self._gripper_close.step()
        self._gripper_open_precondition_node.step()
        self._move_ee.step()
        self._end_effector_control.step()
        self._gripper_close_precondition_node.step()




def precondition(first_behavior, later_behavior, task_node):
    precondition_node_kernel = Kernel.BoxKernel()
    precondition_node_kernel.set_amplitude(2.5)
    precondition_node = DynamicField.DynamicField([], [], precondition_node_kernel)

    precondition_node_weight = DynamicField.Weight(5.5)
    DynamicField.connect(task_node, precondition_node, [precondition_node_weight])

    precondition_inhibition_node = first_behavior.get_cos_memory_node()
    if (first_behavior.is_reactivating()):
        precondition_inhibition_node = first_behavior.get_cos_node()

    precondition_inhibition_weight = DynamicField.Weight(-5.5)
    DynamicField.connect(precondition_inhibition_node, precondition_node, [precondition_inhibition_weight])

    intention_inhibition_weight = DynamicField.Weight(-6.5)
    DynamicField.connect(precondition_node, later_behavior.get_intention_node(), [intention_inhibition_weight])

    return precondition_node

def competition(behavior_0, behavior_1, task_node, bidirectional=False):
    competition_nodes = []
    competition_node_01_kernel = Kernel.BoxKernel()
    competition_node_01_kernel.set_amplitude(1.5)
    competition_node_01 = DynamicField.DynamicField([], [], competition_node_01_kernel)
    competition_nodes.append(competition_node_01)

    competition_node_01_weight = DynamicField.Weight(2.5)
    DynamicField.connect(task_node, competition_node_01, [competition_node_01_weight])

    competition_01_excitation_weight = DynamicField.Weight(2.5)
    DynamicField.connect(behavior_0.get_intention_node(), competition_node_01, [competition_01_excitation_weight])

    intention_1_inhibition_weight = DynamicField.Weight(-5.5)
    DynamicField.connect(competition_node_01, behavior_1.get_intention_node(), [intention_1_inhibition_weight])

    competition_node_10 = None

    if (bidirectional is True):
        competition_node_10_kernel = Kernel.BoxKernel()
        competition_node_10_kernel.set_amplitude(1.5)
        competition_node_10 = DynamicField.DynamicField([], [], competition_node_10_kernel)
        competition_nodes.append(competition_node_10)

        competition_node_10_weight = DynamicField.Weight(2.5)
        DynamicField.connect(task_node, competition_node_10, [competition_node_10_weight])

        competition_10_excitation_weight = DynamicField.Weight(2.5)
        DynamicField.connect(behavior_1.get_intention_node(), competition_node_10, [competition_10_excitation_weight])

        intention_0_inhibition_weight = DynamicField.Weight(-5.5)
        DynamicField.connect(competition_node_10, behavior_0.get_intention_node(), [intention_0_inhibition_weight])

        competition_01_inhibition_weight = DynamicField.Weight(-5.5)
        competition_10_inhibition_weight = DynamicField.Weight(-5.5)
        
        DynamicField.connect(competition_node_01, competition_node_10, [competition_10_inhibition_weight])
        DynamicField.connect(competition_node_10, competition_node_01, [competition_01_inhibition_weight])

    return competition_nodes

def connect_to_task(task, behavior):
    intention_weight = DynamicField.Weight(5.5)
    cos_memory_weight = DynamicField.Weight(2.5)
    DynamicField.connect(task, behavior.get_intention_node(), [intention_weight])
    DynamicField.connect(task, behavior.get_cos_memory_node(), [cos_memory_weight])


class ElementaryBehavior:

    def __init__(self,
                 intention_field,
                 cos_field,
                 int_node_to_int_field_weight,
                 int_node_to_cos_node_weight = None,
                 cos_field_to_cos_node_weight = None,
                 cos_node_to_cos_memory_node_weight = None,
                 int_inhibition_weight = None,
                 reactivating = False,
                 log_activation = False,
                 step_fields = False,
                 name = ""):

        if (int_node_to_cos_node_weight is None):
            int_node_to_cos_node_weight = 2.0
        if (cos_field_to_cos_node_weight is None):
            cos_field_to_cos_node_weight = 3.0
        if (cos_node_to_cos_memory_node_weight is None):
            cos_node_to_cos_memory_node_weight = 2.5
        if (int_inhibition_weight is None):
            int_inhibition_weight = -6.0

        # name of this elementary behavior
        self._name = name

        # intention and cos field
        self._intention_field = intention_field
        self._intention_field.set_name(self._name + " intention field")
        if (log_activation):
            self._intention_field.start_activation_log()
        self._cos_field = cos_field
        self._cos_field.set_name(self._name + " cos field")
        if (log_activation):
            self._cos_field.start_activation_log()

        # connectables that describe weights between different nodes/fields
        self._int_node_to_int_field_weight = DynamicField.Weight(int_node_to_int_field_weight)
        self._int_node_to_cos_node_weight = DynamicField.Weight(int_node_to_cos_node_weight)
        self._cos_field_to_cos_node_weight = DynamicField.Weight(cos_field_to_cos_node_weight)
        self._cos_node_to_cos_memory_node_weight = DynamicField.Weight(cos_node_to_cos_memory_node_weight)
        self._int_inhibition_weight = DynamicField.Weight(int_inhibition_weight)

        # does the node reactivate its intention, when the CoS node gets deactivated?
        self._reactivating = reactivating

        # should the intention and CoS field be stepped, when the elementary
        # behavior is stepped?
        # if the fields belong to other elementary behaviors as well, make sure
        # that they are only stepped once every iteration
        self._step_fields = step_fields

        # intention node and its kernel
        intention_node_kernel = Kernel.BoxKernel()
        intention_node_kernel.set_amplitude(2.5)
        self._intention_node = DynamicField.DynamicField([], [], intention_node_kernel)
        self._intention_node.set_name(self._name + " intention node")
        if (log_activation):
            self._intention_node.start_activation_log()
        # CoS node and its kernel
        cos_node_kernel = Kernel.BoxKernel()
        cos_node_kernel.set_amplitude(2.5)
        self._cos_node = DynamicField.DynamicField([], [], cos_node_kernel)
        self._cos_node.set_name(self._name + " cos node")
        if (log_activation):
            self._cos_node.start_activation_log()
        # CoS memory node and its kernel
        cos_memory_node_kernel = Kernel.BoxKernel()
        cos_memory_node_kernel.set_amplitude(4.5)
        self._cos_memory_node = DynamicField.DynamicField([], [], cos_memory_node_kernel)
        self._cos_memory_node.set_name(self._name + " cos memory node")
        if (log_activation):
            self._cos_memory_node.start_activation_log()

        # connect all connectables in this elementary behavior
        self._connect()

    @classmethod
    def with_internal_fields(cls,
                             field_dimensionality,
                             field_sizes,
                             field_resolutions,
                             int_node_to_int_field_weight,
                             int_node_to_cos_node_weight = None,
                             int_field_to_cos_field_weight = None,
                             cos_field_to_cos_node_weight = None,
                             cos_node_to_cos_memory_node_weight = None,
                             int_inhibition_weight = None,
                             reactivating = False,
                             log_activation = False,
                             name = ""):

        if (int_field_to_cos_field_weight is None):
            int_field_to_cos_field_weight = 4.0

        # intention field and its kernel
        intention_field_kernel = Kernel.GaussKernel(field_dimensionality)
        intention_field_kernel.add_mode(5.0, [1.0] * field_dimensionality, [0.0] * field_dimensionality)
        intention_field_kernel.calculate()
        intention_field = DynamicField.DynamicField(field_sizes, field_resolutions, intention_field_kernel)
        intention_field.set_global_inhibition(100.0)

        # CoS field and its kernel
        cos_field_kernel = Kernel.GaussKernel(field_dimensionality)
        cos_field_kernel.add_mode(5.0, [1.0] * field_dimensionality, [0.0] * field_dimensionality)
        cos_field_kernel.calculate()
        cos_field = DynamicField.DynamicField(field_sizes, field_resolutions, cos_field_kernel)
        cos_field.set_global_inhibition(100.0)

        # connect intention field to cos field
        weight = DynamicField.Weight(int_field_to_cos_field_weight)
        DynamicField.connect(intention_field, cos_field, [weight])

        return cls(intention_field,
                   cos_field,
                   int_node_to_int_field_weight,
                   int_node_to_cos_node_weight,
                   cos_field_to_cos_node_weight,
                   cos_node_to_cos_memory_node_weight,
                   int_inhibition_weight,
                   reactivating,
                   log_activation,
                   step_fields=True,
                   name=name)
 
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

    def is_reactivating(self):
        return self._reactivating

    def _connect(self):
        # connect intention node to intention field
        self._intention_projection = DynamicField.Projection(0, self._intention_field.get_dimensionality(), set([]), [])
        intention_processing_steps = [self._intention_projection, self._int_node_to_int_field_weight]
        DynamicField.connect(self._intention_node, self._intention_field, intention_processing_steps)

        # connect intention node to cos node
        DynamicField.connect(self._intention_node, self._cos_node, [self._int_node_to_cos_node_weight])

        # connect cos field to cos node
        self._cos_projection = DynamicField.Projection(self._cos_field.get_dimensionality(), 0, set([]), [])
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
                        self._cos_node,
                        self._cos_memory_node]

        if (self._step_fields):
            connectables.insert(1, self._intention_field)
            connectables.insert(2, self._cos_field)

        for connectable in connectables:
            connectable.step()

