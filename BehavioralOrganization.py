import math
import random
import Kernel
import numpy
import copy
import DynamicField
import math_tools
import CameraField
import HeadSensorField
import HeadControl
import EndEffectorControl
import GripperControl
import GripperSensor


class GraspArchitecture():

    def __init__(self):
        self.fields = []

        ###############################################################################################################
        ########## ELEMENTARY BEHAVIORS
        ###############################################################################################################


        ###############################################################################################################
        # FIND COLOR (RED OBJECT)
        ###############################################################################################################

        self._find_color_field_size = 15
        find_color_int_weight = math_tools.gauss_1d(self._find_color_field_size, amplitude=15.0, sigma=0.5, shift=0.0)

        self._find_color = ElementaryBehavior.with_internal_fields(field_dimensionality=1,
                                                    field_sizes=[[self._find_color_field_size]],
                                                    field_resolutions=[],
                                                    int_node_to_int_field_weight=find_color_int_weight,
                                                    cos_field_to_cos_node_weight=0,# NEEDS TO BE CHANGED BACK
                                                    name="find color")

        self.fields.append(self._find_color.get_intention_field())
        self.fields.append(self._find_color.get_cos_field())


        ###############################################################################################################
        # FIND COLOR (GREEN EE MARKERS)
        ###############################################################################################################

        # create elementary behavior: find color ee marker
        self._find_color_ee_field_size = 15
        find_color_ee_int_weight = math_tools.gauss_1d(self._find_color_ee_field_size, amplitude=15.0, sigma=0.5, shift=6.0)

        self._find_color_ee = ElementaryBehavior.with_internal_fields(field_dimensionality=1,
                                                    field_sizes=[[self._find_color_ee_field_size]],
                                                    field_resolutions=[],
                                                    int_node_to_int_field_weight=find_color_ee_int_weight,
                                                    cos_field_to_cos_node_weight=0,# NEEDS TO BE CHANGED BACK
                                                    name="find color ee")

        self.fields.append(self._find_color_ee.get_intention_field())
        self.fields.append(self._find_color_ee.get_cos_field())


        ###############################################################################################################
        # MOVE HEAD
        ###############################################################################################################

        # create elementary behavior: move head
        move_head_field_dimensionality = 2
        self._move_head_field_sizes = [40, 30]

        # move head intention field and its kernel
        intention_field_kernel = Kernel.GaussKernel(14.0, [3.0] * move_head_field_dimensionality)
        self._move_head_intention_field = DynamicField.DynamicField([[self._move_head_field_sizes[0]],[self._move_head_field_sizes[1]]], [], [intention_field_kernel])
        self._move_head_intention_field.set_global_inhibition(240.0)
        self._move_head_intention_field.set_relaxation_time(2.0)
        self._move_head_intention_field.set_name("move_head_intention_field")
        self.fields.append(self._move_head_intention_field)

        # move_head CoS field and its kernel
        cos_field_kernel = Kernel.GaussKernel(17.0, [3.0] * move_head_field_dimensionality)
        self._move_head_cos_field = DynamicField.DynamicField([[self._move_head_field_sizes[0]],[self._move_head_field_sizes[1]]], [], [cos_field_kernel])
        self._move_head_cos_field.set_global_inhibition(160.0)
        self._move_head_cos_field.set_relaxation_time(2.0)
        self._move_head_cos_field.set_name("move_head_cos_field")
        self.fields.append(self._move_head_cos_field)

        # create elementary behavior: move head
        move_head_int_weight = numpy.ones((self._move_head_field_sizes)) * 4.0

        self._move_head = ElementaryBehavior(intention_field=self._move_head_intention_field,
                                             cos_field=self._move_head_cos_field,
                                             int_node_to_int_field_weight=move_head_int_weight,
                                             name="move head",
                                             step_fields=True)

        # connect the move head intention and CoS field
        move_head_int_field_to_cos_field_weight = DynamicField.Weight(4.0)
        DynamicField.connect(self._move_head_intention_field, self._move_head_cos_field, [move_head_int_field_to_cos_field_weight])

        # connect move head intention node to its cos field, so that the peak
        # forms in the center of the cos field (cos for the move-head behavior)
        int_node_to_cos_field_projection = DynamicField.Projection(0, move_head_field_dimensionality, set([]), [])
        weight = math_tools.gauss_2d(self._move_head_field_sizes, amplitude=4.2, sigmas=[1.5, 1.5], shifts=[self._move_head_field_sizes[0]/2., self._move_head_field_sizes[1]/2.])
        int_node_to_cos_field_weight = DynamicField.Weight(weight)
        DynamicField.connect(self._move_head.get_intention_node(), self._move_head_cos_field, [int_node_to_cos_field_projection, int_node_to_cos_field_weight])


        ###############################################################################################################
        # MOVE RIGHT ARM
        ###############################################################################################################

        # create elementary behavior: move right arm
        move_arm_field_dimensionality = 2
        self._move_arm_field_sizes = [40, 40]

        # move right arm intention field and its kernel
        intention_field_kernel = Kernel.GaussKernel(10.0, [3.0] * move_arm_field_dimensionality)
        self._move_right_arm_intention_field = DynamicField.DynamicField([[self._move_arm_field_sizes[0]],[self._move_arm_field_sizes[1]]], [], [intention_field_kernel])
        self._move_right_arm_intention_field.set_global_inhibition(60.0)
        self._move_right_arm_intention_field.set_relaxation_time(2.0)
        self._move_right_arm_intention_field.set_name("move_right_arm_intention_field")
        self.fields.append(self._move_right_arm_intention_field)

#        self._move_right_arm_height_intention_field = DynamicField.DynamicField([[15]], [], None)
#        self._move_right_arm_height_intention_field.set_global_inhibition(60.0)
#        self._move_right_arm_height_intention_field.set_relaxation_time(2.0)
#        self._move_right_arm_height_intention_field.set_name("move_right_arm_height_intention_field")
#        self.fields.append(self._move_right_height_arm_intention_field)

#        self._move_right_arm_orient_intention_field = DynamicField.DynamicField([[15]], [], None)
#        self._move_right_arm_orient_intention_field.set_global_inhibition(60.0)
#        self._move_right_arm_orient_intention_field.set_relaxation_time(2.0)
#        self._move_right_arm_orient_intention_field.set_name("move_right_arm_orient_intention_field")
#        self.fields.append(self._move_right_arm_orient_intention_field)


        # move arm CoS field and its kernel
        move_arm_cos_field_dimensionality = move_head_field_dimensionality
        self._move_arm_cos_field_sizes = self._move_head_field_sizes
        cos_field_kernel = Kernel.GaussKernel(10.0, [3.0] * move_arm_cos_field_dimensionality)
        self._move_arm_cos_field = DynamicField.DynamicField([[self._move_arm_cos_field_sizes[0]],[self._move_arm_cos_field_sizes[1]]], [], [cos_field_kernel])
        self._move_arm_cos_field.set_global_inhibition(60.0)
        self._move_arm_cos_field.set_relaxation_time(2.0)
        self._move_arm_cos_field.set_name("move_arm_cos_field")
        self.fields.append(self._move_arm_cos_field)

        # create elementary behavior: move arm
        move_right_arm_int_weight = numpy.ones(self._move_arm_field_sizes) * 2.0

        self._move_right_arm = ElementaryBehavior(intention_field=self._move_right_arm_intention_field,
                                             cos_field=self._move_arm_cos_field,
                                             int_node_to_int_field_weight=move_right_arm_int_weight,
                                             name="move right arm",
                                             step_fields=True)

        # connect move right arm intention node to its cos field, so that the peak
        # forms in the center of the cos field (cos for the move-right-arm behavior)
        int_node_to_cos_field_projection = DynamicField.Projection(0, move_head_field_dimensionality, set([]), [])
        weight = math_tools.gauss_2d(self._move_head_field_sizes, amplitude=2.0, sigmas=[2.0, 2.0], shifts=[self._move_head_field_sizes[0]/2., self._move_head_field_sizes[1]/2.])
        int_node_to_cos_field_weight = DynamicField.Weight(weight)
        DynamicField.connect(self._move_right_arm.get_intention_node(), self._move_arm_cos_field, [int_node_to_cos_field_projection, int_node_to_cos_field_weight])

#        projection = DynamicField.Projection(0, self._move_arm_height_intention_field.get_dimensionality(), set([]), [])
#        weight = math_tools.gauss_1d(self._move_arm_height_intention_field.get_output_sizes()[0], amplitude=6.0, sigma=1.0, shift=0.37)
#        DynamicField.connect(self._move_right_arm.get_intention_node(), self._move_right_arm_height_intention_field, [projection, weight])


        ###############################################################################################################
        # MOVE LEFT ARM
        ###############################################################################################################

        # move left arm intention field and its kernel
        intention_field_kernel = Kernel.GaussKernel(10.0, [3.0] * move_arm_field_dimensionality)
        self._move_left_arm_intention_field = DynamicField.DynamicField([[self._move_arm_field_sizes[0]],[self._move_arm_field_sizes[1]]], [], [intention_field_kernel])
        self._move_left_arm_intention_field.set_global_inhibition(160.0)
        self._move_left_arm_intention_field.set_relaxation_time(2.0)
        self._move_left_arm_intention_field.set_name("move_left_arm_intention_field")
        self.fields.append(self._move_left_arm_intention_field)

        # create elementary behavior: move left arm
        move_left_arm_int_weight = numpy.ones(self._move_arm_field_sizes) * 2.0

        self._move_left_arm = ElementaryBehavior(intention_field=self._move_left_arm_intention_field,
                                             cos_field=self._move_arm_cos_field,
                                             int_node_to_int_field_weight=move_left_arm_int_weight,
                                             name="move left arm",
                                             step_fields=True)

        # connect move arm intention node to its cos field, so that the peak
        # forms in the center of the cos field (cos for the move-left-arm behavior)
        int_node_to_cos_field_projection = DynamicField.Projection(0, move_head_field_dimensionality, set([]), [])
        weight = math_tools.gauss_2d(self._move_head_field_sizes, amplitude=2.0, sigmas=[2.0, 2.0], shifts=[self._move_head_field_sizes[0]/2., self._move_head_field_sizes[1]/2.])
        int_node_to_cos_field_weight = DynamicField.Weight(weight)
        DynamicField.connect(self._move_left_arm.get_intention_node(), self._move_arm_cos_field, [int_node_to_cos_field_projection, int_node_to_cos_field_weight])


        ###############################################################################################################
        # GRIPPER LEFT
        ###############################################################################################################

        # create gripper intention and cos fields
        gripper_field_dimensionality = 1
        self._gripper_field_size = 15

        # gripper left intention field and its kernel
        intention_field_kernel = Kernel.GaussKernel(15.0, [1.0] * gripper_field_dimensionality)
        self._gripper_left_intention_field = DynamicField.DynamicField([[self._gripper_field_size]], [], [intention_field_kernel])
        self._gripper_left_intention_field.set_global_inhibition(50.0)

        # gripper CoS field and its kernel
        cos_field_kernel = Kernel.GaussKernel(15.0, [1.0] * gripper_field_dimensionality)
        self._gripper_left_cos_field = DynamicField.DynamicField([[self._gripper_field_size]], [], [cos_field_kernel])
        self._gripper_left_cos_field.set_global_inhibition(50.0)

        # connect the gripper left intention and CoS field
        gripper_left_int_field_to_cos_field_weight = DynamicField.Weight(2.5)
        DynamicField.connect(self._gripper_left_intention_field, self._gripper_left_cos_field, [gripper_left_int_field_to_cos_field_weight])

        # create elementary behavior: gripper left close
        gripper_left_close_int_weight = math_tools.gauss_1d(self._gripper_field_size, amplitude=15, sigma=0.5, shift=1.0)
        self._gripper_left_close = ElementaryBehavior(intention_field=self._gripper_left_intention_field,
                                                  cos_field=self._gripper_left_cos_field,
                                                  int_node_to_int_field_weight=gripper_left_close_int_weight,
                                                  name="gripper left close")

        # create elementary behavior: gripper open
        gripper_left_open_int_weight = math_tools.gauss_1d(self._gripper_field_size, amplitude=15, sigma=0.5, shift=self._gripper_field_size-1.0)
        self._gripper_left_open = ElementaryBehavior(intention_field=self._gripper_left_intention_field,
                                                  cos_field=self._gripper_left_cos_field,
                                                  int_node_to_int_field_weight=gripper_left_open_int_weight,
                                                  name="gripper left open")
        self._gripper_left_intention_field.set_name("gripper_left_intention_field")
        self._gripper_left_cos_field.set_name("gripper_left_cos_field")

        self.fields.append(self._gripper_left_intention_field)
        self.fields.append(self._gripper_left_cos_field)


        ###############################################################################################################
        # GRIPPER RIGHT
        ###############################################################################################################

        # gripper right intention field and its kernel
        intention_field_kernel = Kernel.GaussKernel(15.0, [1.0] * gripper_field_dimensionality)
        self._gripper_right_intention_field = DynamicField.DynamicField([[self._gripper_field_size]], [], [intention_field_kernel])
        self._gripper_right_intention_field.set_global_inhibition(50.0)

        # gripper CoS field and its kernel
        cos_field_kernel = Kernel.GaussKernel(15.0, [1.0] * gripper_field_dimensionality)
        self._gripper_right_cos_field = DynamicField.DynamicField([[self._gripper_field_size]], [], [cos_field_kernel])
        self._gripper_right_cos_field.set_global_inhibition(50.0)

        # connect the gripper right intention and CoS field
        gripper_right_int_field_to_cos_field_weight = DynamicField.Weight(2.5)
        DynamicField.connect(self._gripper_right_intention_field, self._gripper_right_cos_field, [gripper_right_int_field_to_cos_field_weight])

        # create elementary behavior: gripper right close
        gripper_right_close_int_weight = math_tools.gauss_1d(self._gripper_field_size, amplitude=15, sigma=0.5, shift=1.0)
        self._gripper_right_close = ElementaryBehavior(intention_field=self._gripper_right_intention_field,
                                                  cos_field=self._gripper_right_cos_field,
                                                  int_node_to_int_field_weight=gripper_right_close_int_weight,
                                                  name="gripper right close")

        # create elementary behavior: gripper open
        gripper_right_open_int_weight = math_tools.gauss_1d(self._gripper_field_size, amplitude=15, sigma=0.5, shift=self._gripper_field_size-1.0)
        self._gripper_right_open = ElementaryBehavior(intention_field=self._gripper_right_intention_field,
                                                  cos_field=self._gripper_right_cos_field,
                                                  int_node_to_int_field_weight=gripper_right_open_int_weight,
                                                  name="gripper right open")
        self._gripper_right_intention_field.set_name("gripper_right_intention_field")
        self._gripper_right_cos_field.set_name("gripper_right_cos_field")

        self.fields.append(self._gripper_right_intention_field)
        self.fields.append(self._gripper_right_cos_field)


        ###############################################################################################################
        ########## PERCEPTION LAYER
        ###############################################################################################################


        ###############################################################################################################
        # COLOR SPACE FIELD (RED OBJECT)
        ###############################################################################################################

        # create perception color-space field
        color_space_field_dimensionality = 3
        color_space_kernel = Kernel.GaussKernel(10.0, [3.0] * color_space_field_dimensionality)

        self._color_space_field_sizes = [self._move_head_field_sizes[0], self._move_head_field_sizes[1], self._find_color_field_size]
        self._color_space_field = DynamicField.DynamicField([[self._color_space_field_sizes[0]],[self._color_space_field_sizes[1]],[self._color_space_field_sizes[2]]], [], [color_space_kernel])
        self._color_space_field.set_global_inhibition(50.0)
        self._color_space_field.set_relaxation_time(2.0)
        self._color_space_field.set_name("color_space_field")
        self.fields.append(self._color_space_field)

        fc_int_to_color_space_projection = DynamicField.Projection(self._find_color.get_intention_field().get_dimensionality(), color_space_field_dimensionality, set([0]), [2])
        fc_int_to_color_space_weight = DynamicField.Weight(3.0)
        DynamicField.connect(self._find_color.get_intention_field(), self._color_space_field, [fc_int_to_color_space_weight, fc_int_to_color_space_projection])

        color_space_to_fc_cos_projection = DynamicField.Projection(color_space_field_dimensionality, self._find_color.get_cos_field().get_dimensionality(), set([2]), [0])
        color_space_to_fc_cos_weight = DynamicField.Weight(8.0)
        DynamicField.connect(self._color_space_field, self._find_color.get_cos_field(), [color_space_to_fc_cos_projection, color_space_to_fc_cos_weight])


        ###############################################################################################################
        # COLOR SPACE FIELD (GREEN EE MARKERS)
        ###############################################################################################################

        # create perception color-space field
        color_space_ee_field_dimensionality = 3
        color_space_ee_kernel = Kernel.GaussKernel(10.0, [3.0] * color_space_ee_field_dimensionality)

        self._color_space_ee_field_sizes = [self._move_head_field_sizes[0], self._move_head_field_sizes[1], self._find_color_ee_field_size]
        self._color_space_ee_field = DynamicField.DynamicField([[self._color_space_ee_field_sizes[0]],[self._color_space_ee_field_sizes[1]],[self._color_space_ee_field_sizes[2]]], [], [color_space_ee_kernel])
        self._color_space_ee_field.set_global_inhibition(50.0)
        self._color_space_ee_field.set_relaxation_time(2.0)
        self._color_space_ee_field.set_name("color_space_ee_field")
        self.fields.append(self._color_space_ee_field)

        fc_int_to_color_space_ee_projection = DynamicField.Projection(self._find_color_ee.get_intention_field().get_dimensionality(), color_space_ee_field_dimensionality, set([0]), [2])
        fc_int_to_color_space_ee_weight = DynamicField.Weight(3.0)
        DynamicField.connect(self._find_color_ee.get_intention_field(), self._color_space_ee_field, [fc_int_to_color_space_ee_weight, fc_int_to_color_space_ee_projection])

        color_space_ee_to_fc_cos_projection = DynamicField.Projection(color_space_ee_field_dimensionality, self._find_color_ee.get_cos_field().get_dimensionality(), set([2]), [0])
        color_space_ee_to_fc_cos_weight = DynamicField.Weight(8.0)
        DynamicField.connect(self._color_space_ee_field, self._find_color_ee.get_cos_field(), [color_space_ee_to_fc_cos_projection, color_space_ee_to_fc_cos_weight])


        color_space_ee_to_move_arm_cos_projection = DynamicField.Projection(color_space_ee_field_dimensionality, self._move_left_arm.get_cos_field().get_dimensionality(), set([0,1]), [0,1])
        color_space_ee_to_move_arm_cos_weight = DynamicField.Weight(4.0)
        DynamicField.connect(self._color_space_ee_field, self._move_left_arm.get_cos_field(), [color_space_ee_to_move_arm_cos_weight, color_space_ee_to_move_arm_cos_projection])


        ###############################################################################################################
        # SPATIAL TARGET FOR EE
        ###############################################################################################################

        # create "spatial target location" field
        spatial_target_field_dimensionality = 2
        spatial_target_kernel = Kernel.GaussKernel(13.0, [3.0] * spatial_target_field_dimensionality)

        self._spatial_target_field_sizes = self._move_head_field_sizes
        self._spatial_target_field = DynamicField.DynamicField([[self._spatial_target_field_sizes[0]], [self._spatial_target_field_sizes[1]]], [], [spatial_target_kernel])
        self._spatial_target_field.set_global_inhibition(150.0)
        self._spatial_target_field.set_relaxation_time(2.0)
        self._spatial_target_field.set_name("spatial_target_field")
        self.fields.append(self._spatial_target_field)

        color_space_to_spatial_target_projection = DynamicField.Projection(color_space_field_dimensionality, spatial_target_field_dimensionality, set([0, 1]), [0, 1])
        color_space_to_spatial_target_weight = DynamicField.Weight(9.0)
        DynamicField.connect(self._color_space_field, self._spatial_target_field, [color_space_to_spatial_target_projection, color_space_to_spatial_target_weight])

        spatial_target_to_move_head_int_weight = DynamicField.Weight(4.0)
        DynamicField.connect(self._spatial_target_field, self._move_head.get_intention_field(), [spatial_target_to_move_head_int_weight])


        ###############################################################################################################
        # PERCEPTION FIELD
        ###############################################################################################################

        # create perception field in end effector space
#        perception_ee_field_dimensionality = 2
#        perception_ee_kernel = Kernel.GaussKernel(15.0, [1.0] * perception_ee_field_dimensionality)

#        self._perception_ee_field_sizes = self._move_head_field_sizes
#        self._perception_ee_field = DynamicField.DynamicField([[self._perception_ee_field_sizes[0]], [self._perception_ee_field_sizes[1]]], [], [perception_ee_kernel])
#        self._perception_ee_field.set_global_inhibition(400.0)
#        self._perception_ee_field.set_name("perception_ee_field")
#        self.fields.append(self._perception_ee_field)

#        perception_ee_to_move_head_cos_weight = DynamicField.Weight(3.5)
#        DynamicField.connect(self._perception_ee_field, self._move_head.get_cos_field(), [perception_ee_to_move_head_cos_weight])




        ###############################################################################################################
        ########## SENSORY MOTOR LAYER
        ###############################################################################################################


        ###############################################################################################################
        # CAMERA
        ###############################################################################################################

        # create "camera" field
        self._camera_field = CameraField.NaoCameraField()
        self._camera_field.set_name("camera_field")
        self.fields.append(self._camera_field)
        self._camera_field_sizes = self._camera_field.get_output_dimension_sizes()

        camera_to_color_space_weight = DynamicField.Weight(3.0)
        camera_to_color_space_ee_weight = DynamicField.Weight(3.0)
        DynamicField.connect(self._camera_field, self._color_space_field, [camera_to_color_space_weight])
        DynamicField.connect(self._camera_field, self._color_space_ee_field, [camera_to_color_space_ee_weight])


        ###############################################################################################################
        # HEAD CONTROL
        ###############################################################################################################

        # create head control connectable
        self._head_control = HeadControl.HeadControl(self._move_head_field_sizes, head_speed_fraction = 0.3)
        DynamicField.connect(self._move_head.get_intention_field(), self._head_control)


        ###############################################################################################################
        # HEAD SENSOR
        ###############################################################################################################

        # create head sensor field
        self._head_sensor_field = HeadSensorField.NaoHeadSensorField(use_robot_sensors = True)
        self._head_sensor_field.set_name("head_sensor_field")
        self.fields.append(self._head_sensor_field)
        self._head_sensor_field_sizes = self._head_sensor_field.get_output_dimension_sizes()

        head_sensor_to_move_right_arm_int_weight = DynamicField.Weight(4.0)
        DynamicField.connect(self._head_sensor_field, self._move_right_arm_intention_field, [head_sensor_to_move_right_arm_int_weight])
        head_sensor_to_move_left_arm_int_weight = DynamicField.Weight(4.0)
        DynamicField.connect(self._head_sensor_field, self._move_left_arm_intention_field, [head_sensor_to_move_left_arm_int_weight])


        ###############################################################################################################
        # SIDE SENSOR
        ###############################################################################################################

        # node that represents the left side
        self._side_left = DynamicField.DynamicField([], [], None)

        head_sensor_field_to_side_left_projection = DynamicField.Projection(self._head_sensor_field.get_dimensionality(), 0, set([]), [])
        grid_h, grid_v = numpy.mgrid[20:-20:self._head_sensor_field_sizes[0] + 0j,20:-20:self._head_sensor_field_sizes[1] + 0j]
        weight_left = numpy.tanh(20 * grid_h) * 5.2
        head_sensor_field_to_side_left_weight = DynamicField.Weight(weight_left)

        DynamicField.connect(self._head_sensor_field, self._side_left, [head_sensor_field_to_side_left_weight, head_sensor_field_to_side_left_projection])

        # node that represents the right side
        self._side_right = DynamicField.DynamicField([], [], None)

        head_sensor_field_to_side_right_projection = DynamicField.Projection(self._head_sensor_field.get_dimensionality(), 0, set([]), [])
        grid_h, grid_v = numpy.mgrid[-20:20:self._head_sensor_field_sizes[0] + 0j,-20:20:self._head_sensor_field_sizes[1] + 0j]
        weight_right = numpy.tanh(20 * grid_h) * 5.2
        head_sensor_field_to_side_right_weight = DynamicField.Weight(weight_right)

        DynamicField.connect(self._head_sensor_field, self._side_right, [head_sensor_field_to_side_right_weight, head_sensor_field_to_side_right_projection])


        ###############################################################################################################
        # EE CONTROL
        ###############################################################################################################

        # create end effector control connectable
        self._end_effector_control_right = EndEffectorControl.NaoEndEffectorControlRight(self._head_sensor_field, self._move_arm_field_sizes, end_effector_speed_fraction = 0.1, use_robot_sensors = True)
        DynamicField.connect(self._move_right_arm.get_intention_field(), self._end_effector_control_right)
        self._end_effector_control_left = EndEffectorControl.NaoEndEffectorControlLeft(self._head_sensor_field, self._move_arm_field_sizes, end_effector_speed_fraction = 0.1, use_robot_sensors = True)
        DynamicField.connect(self._move_left_arm.get_intention_field(), self._end_effector_control_left)


        ###############################################################################################################
        # GRIPPER CONTROL
        ###############################################################################################################

        # create gripper control for the right hand
        self._gripper_control_right = GripperControl.NaoGripperControlRight(self._gripper_field_size, gripper_speed_fraction = 1.0, use_robot_sensors = True)
        DynamicField.connect(self._gripper_right_open.get_intention_field(), self._gripper_control_right)

        # create gripper control for the left hand
        self._gripper_control_left = GripperControl.NaoGripperControlLeft(self._gripper_field_size, gripper_speed_fraction = 1.0, use_robot_sensors = True)
        DynamicField.connect(self._gripper_left_open.get_intention_field(), self._gripper_control_left)


        ###############################################################################################################
        # GRIPPER SENSOR
        ###############################################################################################################

        # create gripper sensor for the right hand
        self._gripper_sensor_right = GripperSensor.NaoGripperSensorRight(self._gripper_field_size, use_robot_sensors = True)
        gripper_sensor_right_weight = DynamicField.Weight(4.0)
        DynamicField.connect(self._gripper_sensor_right, self._gripper_right_open.get_cos_field(), [gripper_sensor_right_weight])

        # create gripper sensor for the left hand
        self._gripper_sensor_left = GripperSensor.NaoGripperSensorLeft(self._gripper_field_size, use_robot_sensors = True)
        gripper_sensor_left_weight = DynamicField.Weight(4.0)
        DynamicField.connect(self._gripper_sensor_left, self._gripper_left_open.get_cos_field(), [gripper_sensor_left_weight])



        ###############################################################################################################
        ########## BEHAVIORAL ORGANIZATION LAYER
        ###############################################################################################################


        ###############################################################################################################
        # TASK NODE
        ###############################################################################################################

        self._task_node = DynamicField.DynamicField([], [], None)
        self._task_node.set_boost(10)


        ###############################################################################################################
        # PRECONDITION NODES
        ###############################################################################################################

        # create precondition nodes
        self._side_left_to_gripper_left_open_precondition = precondition(self._side_left, self._gripper_left_open, self._task_node)
        self._side_right_to_gripper_right_open_precondition = precondition(self._side_right, self._gripper_right_open, self._task_node)

        self._side_left_to_gripper_left_close_precondition = precondition(self._side_left, self._gripper_left_close, self._task_node)
        self._side_right_to_gripper_right_close_precondition = precondition(self._side_right, self._gripper_right_close, self._task_node)

        self._move_head_to_gripper_left_open_precondition = precondition(self._move_head, self._gripper_left_open, self._task_node)
        self._move_head_to_gripper_right_open_precondition = precondition(self._move_head, self._gripper_right_open, self._task_node)

        self._gripper_left_open_to_move_left_arm_precondition = precondition(self._gripper_left_open, self._move_left_arm, self._task_node)
        self._gripper_right_open_to_move_right_arm_precondition = precondition(self._gripper_right_open, self._move_right_arm, self._task_node)

        self._move_left_arm_to_gripper_left_close_precondition = precondition(self._move_left_arm, self._gripper_left_close, self._task_node)
        self._move_right_arm_to_gripper_right_close_precondition = precondition(self._move_right_arm, self._gripper_right_close, self._task_node)

        self._find_color_ee_precondition_node = precondition(self._move_head, self._find_color_ee, self._task_node)


        ###############################################################################################################
        # COMPETITION NODES
        ###############################################################################################################

        self._left_right_competition_nodes = competition(self._side_left, self._side_right, self._task_node, bidirectional=True)


        ###############################################################################################################
        # TASK CONNECTIONS
        ###############################################################################################################

        # connect all elementary behaviors to the task node
        connect_to_task(self._task_node, self._find_color)
        connect_to_task(self._task_node, self._move_head)
        connect_to_task(self._task_node, self._move_right_arm)
        connect_to_task(self._task_node, self._move_left_arm)
        connect_to_task(self._task_node, self._gripper_left_open)
        connect_to_task(self._task_node, self._gripper_left_close)
        connect_to_task(self._task_node, self._gripper_right_open)
        connect_to_task(self._task_node, self._gripper_right_close)
        connect_to_task(self._task_node, self._find_color_ee)


    def step(self):
        self._task_node.step()
        self._camera_field.step()
        self._find_color.step()
        self._find_color_ee.step()
#        self._perception_ee_field.step()
        self._color_space_field.step()
        self._color_space_ee_field.step()
        self._spatial_target_field.step()
        self._gripper_left_intention_field.step()
        self._gripper_left_cos_field.step()
        self._gripper_left_open.step()
#        print("gripper open left int: ", self._gripper_left_open.get_intention_node().get_activation())

        self._gripper_left_close.step()
        self._gripper_right_intention_field.step()
        self._gripper_right_cos_field.step()
        self._gripper_right_open.step()
#        print("gripper open right int: ", self._gripper_right_open.get_intention_node().get_activation())
        self._gripper_right_close.step()
        self._move_head.step()
        self._head_control.step()
        self._move_right_arm.step()
        self._move_left_arm.step()
        self._head_sensor_field.step()
        self._end_effector_control_right.step()
        self._end_effector_control_left.step()
        self._side_left.step()
        self._side_right.step()

#        print("side left: ", self._side_left.get_activation())
#        print("side right: ", self._side_right.get_activation())

        self._left_right_competition_nodes[0].step()
        self._left_right_competition_nodes[1].step()
#        print("left right competition 0: ", self._left_right_competition_nodes[0].get_activation())
#        print("left right competition 1: ", self._left_right_competition_nodes[1].get_activation())

        self._side_left_to_gripper_left_open_precondition.step()
        self._side_right_to_gripper_right_open_precondition.step()
        self._side_left_to_gripper_left_close_precondition.step()
        self._side_right_to_gripper_right_close_precondition.step()

        self._move_head_to_gripper_left_open_precondition.step()
        self._move_head_to_gripper_right_open_precondition.step()

        self._gripper_left_open_to_move_left_arm_precondition.step()
        self._gripper_right_open_to_move_right_arm_precondition.step()

        self._move_left_arm_to_gripper_left_close_precondition.step()
        self._move_right_arm_to_gripper_right_close_precondition.step()

        self._find_color_ee_precondition_node.step()

        self._gripper_sensor_right.step()
        self._gripper_sensor_left.step()



def precondition(first, second, task_node):
    precondition_inhibiting_node = first
    inhibited_node = second

    precondition_node_kernel = Kernel.BoxKernel(amplitude=2.5)
    precondition_node = DynamicField.DynamicField([], [], [precondition_node_kernel])

    precondition_node_weight = DynamicField.Weight(5.5)
    DynamicField.connect(task_node, precondition_node, [precondition_node_weight])

    if (isinstance(first, ElementaryBehavior)):
        precondition_inhibiting_node = first.get_cos_memory_node()
        if (first.is_reactivating()):
            precondition_inhibiting_node = first.get_cos_node()

    if (isinstance(second, ElementaryBehavior)):
        inhibited_node = second.get_intention_node()

    precondition_inhibition_weight = DynamicField.Weight(-5.5)
    DynamicField.connect(precondition_inhibiting_node, precondition_node, [precondition_inhibition_weight])

    intention_inhibition_weight = DynamicField.Weight(-6.5)
    DynamicField.connect(precondition_node, inhibited_node, [intention_inhibition_weight])

    return precondition_node

def competition(first, second, task_node, bidirectional=False):
    node_0 = first
    node_1 = second

    if (isinstance(first, ElementaryBehavior)):
        node_0 = behavior_0.get_intention_node()

    if (isinstance(second, ElementaryBehavior)):
        node_1 = behavior_1.get_intention_node()

    competition_nodes = []
    competition_node_01_kernel = Kernel.BoxKernel(amplitude=1.5)
    competition_node_01 = DynamicField.DynamicField([], [], [competition_node_01_kernel])
    competition_node_01.set_relaxation_time(1.0)
    competition_nodes.append(competition_node_01)

    competition_node_01_weight = DynamicField.Weight(2.5)
    DynamicField.connect(task_node, competition_node_01, [competition_node_01_weight])

    competition_01_excitation_weight = DynamicField.Weight(2.5)
    DynamicField.connect(node_0, competition_node_01, [competition_01_excitation_weight])

    intention_1_inhibition_weight = DynamicField.Weight(-6.0)
    DynamicField.connect(competition_node_01, node_1, [intention_1_inhibition_weight])

    competition_node_10 = None

    if (bidirectional is True):
        competition_node_10_kernel = Kernel.BoxKernel(amplitude=1.5)
        competition_node_10 = DynamicField.DynamicField([], [], [competition_node_10_kernel])
        competition_node_10.set_relaxation_time(1.0)
        competition_nodes.append(competition_node_10)

        competition_node_10_weight = DynamicField.Weight(2.5)
        DynamicField.connect(task_node, competition_node_10, [competition_node_10_weight])

        competition_10_excitation_weight = DynamicField.Weight(2.5)
        DynamicField.connect(node_1, competition_node_10, [competition_10_excitation_weight])

        intention_0_inhibition_weight = DynamicField.Weight(-6.0)
        DynamicField.connect(competition_node_10, node_0, [intention_0_inhibition_weight])

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
        intention_node_kernel = Kernel.BoxKernel(amplitude=2.5)
        self._intention_node = DynamicField.DynamicField([], [], [intention_node_kernel])
        self._intention_node.set_name(self._name + " intention node")
        if (log_activation):
            self._intention_node.start_activation_log()
        # CoS node and its kernel
        cos_node_kernel = Kernel.BoxKernel(amplitude=2.5)
        self._cos_node = DynamicField.DynamicField([], [], [cos_node_kernel])
        self._cos_node.set_name(self._name + " cos node")
        if (log_activation):
            self._cos_node.start_activation_log()
        # CoS memory node and its kernel
        cos_memory_node_kernel = Kernel.BoxKernel(amplitude=4.5)
        self._cos_memory_node = DynamicField.DynamicField([], [], [cos_memory_node_kernel])
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
        intention_field_kernel = Kernel.GaussKernel(5.0, [3.0] * field_dimensionality)
        intention_field = DynamicField.DynamicField(field_sizes, field_resolutions, [intention_field_kernel])
        intention_field.set_global_inhibition(100.0)

        # CoS field and its kernel
        cos_field_kernel = Kernel.GaussKernel(5.0, [3.0] * field_dimensionality)
        cos_field = DynamicField.DynamicField(field_sizes, field_resolutions, [cos_field_kernel])
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

