import naoqi
from naoqi import ALProxy
import numpy
import DynamicField
import math_tools
import math

class EndEffectorControlRight(DynamicField.Connectable):
    "End effector control"

    def __init__(self, head_sensor_field, move_arm_intention_field, visual_servoing_intention_field, input_dimension_sizes, end_effector_speed_fraction = 0.02, use_robot_sensors = True):
        "Constructor"
        DynamicField.Connectable.__init__(self)

        # PLANE
        self._end_effector_speed_fraction = end_effector_speed_fraction
        self._use_robot_sensors = use_robot_sensors
        self._input_dimensionality = 2
        self._input_dimension_sizes = input_dimension_sizes
        self._head_sensor_field = head_sensor_field
        self._move_arm_intention_field = move_arm_intention_field
        self._visual_servoing_intention_field = visual_servoing_intention_field

        self._file = open("right_ee_pos_0.dat", 'w')

        # naoqi proxy to talk to the motion module
        self._motion_proxy = ALProxy("ALMotion", "nao.ini.rub.de", 9559)
        # set the stiffness of the arm to 1.0, so it will move
        self._motion_proxy.setStiffnesses("RArm", 1.0)

        # HEIGHT
        self._target_z = 0.355
        self._target_alpha = math.pi / 2.0

        self._intention_node = DynamicField.DynamicField([], [], None)
        self._intention_node.set_boost(6.0)


    def __del__(self):
        self._motion_proxy.setStiffnesses("RArm", 0.0)
        self._file.close()

    def get_intention_node(self):
        return self._intention_node

    def _step_computation(self):

        self._intention_node.step()
        print("intention node: ", self._intention_node.get_output()[0])

        # MOVE ARM
        move_arm_field_output = self._move_arm_intention_field.get_output()
        move_arm_field_output_x = move_arm_field_output.max(0)
        move_arm_field_output_y = move_arm_field_output.max(1)

        normalization_factor_x = move_arm_field_output_x.sum()
        normalization_factor_y = move_arm_field_output_y.sum()

#        print("normalization factor x: ", normalization_factor_x)
#        print("normalization factor y: ", normalization_factor_y)

        # compute ramps for x and y direction
        move_arm_field_length_x = len(move_arm_field_output_x)
        move_arm_field_length_y = len(move_arm_field_output_y)

        min_x = self._head_sensor_field.get_min_x()
        max_x = self._head_sensor_field.get_max_x()
        min_y = self._head_sensor_field.get_min_y()
        max_y = self._head_sensor_field.get_max_y()

        move_arm_ramp_x = numpy.linspace(min_x, max_x , move_arm_field_length_x)
        move_arm_ramp_y = numpy.linspace(min_y, max_y, move_arm_field_length_y)

#        print("ramp x: ", str(move_arm_ramp_x))
#        print("ramp y: ", str(move_arm_ramp_y))

        # get the force values for x,y towards the peak.
        move_arm_boost_x = numpy.dot(move_arm_field_output_x, move_arm_ramp_x)
        move_arm_boost_y = numpy.dot(move_arm_field_output_y, move_arm_ramp_y)

#        print("boost x: ", str(move_arm_boost_x))
#        print("boost y: ", str(move_arm_boost_y))


        # VISUAL SERVOING
        vis_arm_field_output = self._visual_servoing_intention_field.get_output()
        vis_arm_field_output_x = vis_arm_field_output.max(0)
        vis_arm_field_output_y = vis_arm_field_output.max(1)

        vis_arm_field_length_x = len(vis_arm_field_output_x)
        vis_arm_field_length_y = len(vis_arm_field_output_y)

        vis_arm_ramp_x = numpy.linspace(-1.0, 1.0, vis_arm_field_length_x)
        vis_arm_ramp_y = numpy.linspace(-1.0, 1.0, vis_arm_field_length_y)

        vis_arm_boost_x = numpy.dot(vis_arm_field_output_x, vis_arm_ramp_x) / -400.0
        vis_arm_boost_y = numpy.dot(vis_arm_field_output_y, vis_arm_ramp_y) /  400.0

        print("boost x: ", str(vis_arm_boost_x))
        print("boost y: ", str(vis_arm_boost_y))



        # STEP
        current_pos = self._motion_proxy.getPosition("RArm", 2, True)
        current_x = current_pos[0]
        current_y = current_pos[1]
        current_z = current_pos[2]
        current_alpha = current_pos[3]

        self._file.write("" + str(current_y) + "\t" + str(current_x) + "\n")

        relaxation_time = 0.025
        x_dot = self._intention_node.get_output()[0] * (relaxation_time * (-1 * normalization_factor_x * current_x + move_arm_boost_x + vis_arm_boost_x))
        y_dot = self._intention_node.get_output()[0] * (relaxation_time * (-1 * normalization_factor_y * current_y + move_arm_boost_y + vis_arm_boost_y))
        z_dot = self._intention_node.get_output()[0] * (0.6 * (-1 * current_z + self._target_z))
        alpha_dot = self._intention_node.get_output()[0] * (0.3 * (-1 * current_alpha + self._target_alpha))

        end_effector_change = [x_dot, y_dot, z_dot, alpha_dot, 0.0, 0.0]

        print("right change: ", end_effector_change)

        # move the arm towards the peak
        # (the last parameter is the axis mask and determines, what should be
        # controlled: 7 for position only, 56 for orientation only, and 63
        # for position and orientation
        self._motion_proxy.changePosition("RArm", 2, end_effector_change, self._end_effector_speed_fraction, 15)


class EndEffectorControlLeft(DynamicField.Connectable):
    "End effector control"

    def __init__(self, head_sensor_field, move_arm_intention_field, visual_servoing_intention_field, input_dimension_sizes, end_effector_speed_fraction = 0.02, use_robot_sensors = True):
        "Constructor"
        DynamicField.Connectable.__init__(self)

        # PLANE
        self._end_effector_speed_fraction = end_effector_speed_fraction
        self._use_robot_sensors = use_robot_sensors
        self._input_dimensionality = 2
        self._input_dimension_sizes = input_dimension_sizes
        self._head_sensor_field = head_sensor_field
        self._move_arm_intention_field = move_arm_intention_field
        self._visual_servoing_intention_field = visual_servoing_intention_field

        self._file = open("left_ee_pos_0.dat", 'w')

        # naoqi proxy to talk to the motion module
        self._motion_proxy = ALProxy("ALMotion", "nao.ini.rub.de", 9559)
        # set the stiffness of the arm to 1.0, so it will move
        self._motion_proxy.setStiffnesses("LArm", 1.0)

        # HEIGHT
        self._target_z = 0.355
        self._target_alpha = -math.pi / 2.0

        self._intention_node = DynamicField.DynamicField([], [], None)
        self._intention_node.set_boost(6.0)


    def __del__(self):
        self._motion_proxy.setStiffnesses("LArm", 0.0)
        self._file.close()

    def get_intention_node(self):
        return self._intention_node

    def _step_computation(self):

        self._intention_node.step()

        # MOVE ARM
        move_arm_field_output = self._move_arm_intention_field.get_output()
        move_arm_field_output_x = move_arm_field_output.max(0)
        move_arm_field_output_y = move_arm_field_output.max(1)

        normalization_factor_x = move_arm_field_output_x.sum()
        normalization_factor_y = move_arm_field_output_y.sum()

#        print("normalization factor x: ", normalization_factor_x)
#        print("normalization factor y: ", normalization_factor_y)

        # compute ramps for x and y direction
        move_arm_field_length_x = len(move_arm_field_output_x)
        move_arm_field_length_y = len(move_arm_field_output_y)

        min_x = self._head_sensor_field.get_min_x()
        max_x = self._head_sensor_field.get_max_x()
        min_y = self._head_sensor_field.get_min_y()
        max_y = self._head_sensor_field.get_max_y()

        move_arm_ramp_x = numpy.linspace(min_x, max_x , move_arm_field_length_x)
        move_arm_ramp_y = numpy.linspace(min_y, max_y, move_arm_field_length_y)

#        print("ramp x: ", str(move_arm_ramp_x))
#        print("ramp y: ", str(move_arm_ramp_y))

        # get the force values for x,y towards the peak.
        move_arm_boost_x = numpy.dot(move_arm_field_output_x, move_arm_ramp_x)
        move_arm_boost_y = numpy.dot(move_arm_field_output_y, move_arm_ramp_y)

#        print("boost x: ", str(move_arm_boost_x))
#        print("boost y: ", str(move_arm_boost_y))


        # VISUAL SERVOING
        vis_arm_field_output = self._visual_servoing_intention_field.get_output()
        vis_arm_field_output_x = vis_arm_field_output.max(0)
        vis_arm_field_output_y = vis_arm_field_output.max(1)

        vis_arm_field_length_x = len(vis_arm_field_output_x)
        vis_arm_field_length_y = len(vis_arm_field_output_y)

        vis_arm_ramp_x = numpy.linspace(-1.0, 1.0, vis_arm_field_length_x)
        vis_arm_ramp_y = numpy.linspace(-1.0, 1.0, vis_arm_field_length_y)

        vis_arm_boost_x = numpy.dot(vis_arm_field_output_x, vis_arm_ramp_x) / -400.0
        vis_arm_boost_y = numpy.dot(vis_arm_field_output_y, vis_arm_ramp_y) /  400.0


        # STEP
        current_pos = self._motion_proxy.getPosition("LArm", 2, True)
        current_x = current_pos[0]
        current_y = current_pos[1]
        current_z = current_pos[2]
        current_alpha = current_pos[3]

        self._file.write("" + str(current_y) + "\t" + str(current_x) + "\n")

        relaxation_time = 0.025
        x_dot = self._intention_node.get_output()[0] * (relaxation_time * (-1 * normalization_factor_x * current_x + move_arm_boost_x + vis_arm_boost_x))
        y_dot = self._intention_node.get_output()[0] * (relaxation_time * (-1 * normalization_factor_y * current_y + move_arm_boost_y + vis_arm_boost_y))
        z_dot = self._intention_node.get_output()[0] * (0.6 * (-1 * current_z + self._target_z))
        alpha_dot = self._intention_node.get_output()[0] * (0.6 * (-1 * current_alpha + self._target_alpha))

        end_effector_change = [x_dot, y_dot, z_dot, alpha_dot, 0.0, 0.0]

        print("left change: ", end_effector_change)

        # move the arm towards the peak
        # (the last parameter is the axis mask and determines, what should be
        # controlled: 7 for position only, 56 for orientation only, and 63
        # for position and orientation
        self._motion_proxy.changePosition("LArm", 2, end_effector_change, self._end_effector_speed_fraction, 15)


