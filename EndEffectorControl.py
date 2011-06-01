import naoqi
from naoqi import ALProxy
import numpy
import DynamicField
import math_tools
import math

class NaoEndEffectorControl(DynamicField.Connectable):
    "End effector control"

    def __init__(self, head_sensor_field, input_dimension_sizes, end_effector_speed_fraction = 0.2, use_robot_sensors = False):
        "Constructor"
        DynamicField.Connectable.__init__(self)
        self._end_effector_speed_fraction = end_effector_speed_fraction
        self._use_robot_sensors = use_robot_sensors
        self._input_dimensionality = 2
        self._input_dimension_sizes = input_dimension_sizes
        self._head_sensor_field = head_sensor_field

        initial_x = 0.05
        initial_y = -0.10
        initial_z = 0.25

        # create nodes controlling the end effector
        self._end_effector_x = DynamicField.DynamicField([], [], None)
        self._end_effector_x.set_resting_level(0.)
        self._end_effector_x.set_initial_activation(initial_x)
        self._end_effector_x.set_noise_strength(0.0)
        self._end_effector_x.set_relaxation_time(60.)
        self._end_effector_y = DynamicField.DynamicField([], [], None)
        self._end_effector_y.set_resting_level(0.)
        self._end_effector_y.set_initial_activation(initial_y)
        self._end_effector_y.set_noise_strength(0.0)
        self._end_effector_y.set_relaxation_time(60.)

        # naoqi proxy to talk to the motion module
        self._motion_proxy = ALProxy("ALMotion", "192.168.0.102", 9559)
        # set the stiffness of the arm to 1.0, so it will move
        self._motion_proxy.setStiffnesses("RArm", 1.0)
        # move the end effector to an initial position to make grasping
        # easier
        # [position x, pos y, pos z, orientation alpha, ori beta, ori gamma]
        initial_end_effector_configuration = [initial_x, initial_y, initial_z, 0.0, 0.0, 0.0]
        self._motion_proxy.positionInterpolation("RArm", 2, initial_end_effector_configuration, 7, 3, True)

    def __del__(self):
        self._motion_proxy.setStiffnesses("RArm", 0.0)

    def _step_computation(self):
        # extract x and y position of peak
        # the x and y coordinates are switched in the field, in relation to the
        # robot coordinate system
        int_field_output = self.get_incoming_connectables()[0].get_output()
        int_field_output_y = int_field_output.max(1)
        int_field_output_x = int_field_output.max(0)

        # set the normalization factor of the end effector nodes
        self._end_effector_x.set_normalization_factor(int_field_output_x.sum())
        self._end_effector_y.set_normalization_factor(int_field_output_y.sum())

        # compute ramps for x and y direction
        field_length_x = len(int_field_output_x)
        field_length_y = len(int_field_output_y)

        min_x = self._head_sensor_field.get_min_x()
        max_x = self._head_sensor_field.get_max_x()
        min_y = self._head_sensor_field.get_min_y()
        max_y = self._head_sensor_field.get_max_y()

        ramp_x = numpy.linspace(min_x, max_x , field_length_x)
        ramp_y = numpy.linspace(min_y, max_y, field_length_y)

        # get the force values for x,y towards the peak.
        end_effector_x_boost = numpy.dot(int_field_output_x, ramp_x)
        end_effector_y_boost = -1 * numpy.dot(int_field_output_y, ramp_y)

        current_x = self._motion_proxy.getPosition("RArm", 2, True)[0]
        current_y = self._motion_proxy.getPosition("RArm", 2, True)[1]
        self._end_effector_x.set_boost(end_effector_x_boost)
        self._end_effector_y.set_boost(end_effector_y_boost)

        x_dot = self._end_effector_x.get_change(current_x)
        y_dot = self._end_effector_y.get_change(current_y)

        print("x dot: ", str(x_dot))
        print("y dot: ", str(y_dot))

        # compute the change values for x,y of the end_effector
        end_effector_change = [x_dot,
                               y_dot,
                               0., # position z
                               0., # orientation alpha
                               0., # orientation beta
                               0.] # orientation gamma

        # move the arm towards the peak
        # (the last parameter is the axis mask and determines, what should be
        # controlled: 7 for position only, 56 for orientation only, and 63
        # for position and orientation
        #self._motion_proxy.changePosition("RArm", 0, end_effector_change, self._end_effector_speed_fraction, 7)
