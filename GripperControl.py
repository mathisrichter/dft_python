import naoqi
from naoqi import ALProxy
import numpy
import DynamicField
import math_tools
import math

class NaoGripperControlRight(DynamicField.Connectable):
    "Gripper control right"

    def __init__(self, input_dimension_size, gripper_speed_fraction = 0.2, use_robot_sensors = True):
        "Constructor"
        DynamicField.Connectable.__init__(self)
        self._gripper_speed_fraction = gripper_speed_fraction
        self._use_robot_sensors = use_robot_sensors
        self._input_dimensionality = 1
        self._input_dimension_sizes = [input_dimension_size]

        initial_x = 0.00

        # create nodes controlling the end effector
        self._gripper_x = DynamicField.DynamicField([], [], None)
        self._gripper_x.set_resting_level(0.)
        self._gripper_x.set_initial_activation(initial_x)
        self._gripper_x.set_noise_strength(0.0)
        self._gripper_x.set_relaxation_time(5.)


        # naoqi proxy to talk to the motion module
        self._motion_proxy = ALProxy("ALMotion", "192.168.0.102", 9559)
        # set the stiffness of the hand to 1.0, so it will move
        self._motion_proxy.setStiffnesses("RHand", 1.0)

    def __del__(self):
        self._motion_proxy.setStiffnesses("RHand", 0.0)

    def _step_computation(self):
        # extract x position of peak
        int_field_output = self.get_incoming_connectables()[0].get_output()

        # set the normalization factor of the end effector nodes
        self._gripper_x.set_normalization_factor(int_field_output.sum())

        # compute ramps for x and y direction
        field_length_x = len(int_field_output)

        min_x = 0
        max_x = 1.0

        ramp_x = numpy.linspace(min_x, max_x , field_length_x)

        # get the force values for x towards the peak.
        gripper_x_boost = numpy.dot(int_field_output, ramp_x)

        #print("boost x: ", str(gripper_x_boost))

        current_x = self._motion_proxy.getAngles("RHand", True)[0]
        self._gripper_x.set_boost(gripper_x_boost)

        x_dot = self._gripper_x.get_change(current_x)[0]

        #print("current right hand angle: ", str(current_x))
        #print("x dot: ", str(x_dot))

        # move the hand towards the peak
        self._motion_proxy.changeAngles("RHand", x_dot, self._gripper_speed_fraction)

class NaoGripperControlLeft(DynamicField.Connectable):
    "Gripper control left"

    def __init__(self, input_dimension_size, gripper_speed_fraction = 0.2, use_robot_sensors = True):
        "Constructor"
        DynamicField.Connectable.__init__(self)
        self._gripper_speed_fraction = gripper_speed_fraction
        self._use_robot_sensors = use_robot_sensors
        self._input_dimensionality = 1
        self._input_dimension_sizes = [input_dimension_size]

        initial_x = 0.00

        # create nodes controlling the end effector
        self._gripper_x = DynamicField.DynamicField([], [], None)
        self._gripper_x.set_resting_level(0.)
        self._gripper_x.set_initial_activation(initial_x)
        self._gripper_x.set_noise_strength(0.0)
        self._gripper_x.set_relaxation_time(5.)


        # naoqi proxy to talk to the motion module
        self._motion_proxy = ALProxy("ALMotion", "192.168.0.102", 9559)
        # set the stiffness of the hand to 1.0, so it will move
        self._motion_proxy.setStiffnesses("LHand", 1.0)

    def __del__(self):
        self._motion_proxy.setStiffnesses("LHand", 0.0)

    def _step_computation(self):
        # extract x position of peak
        int_field_output = self.get_incoming_connectables()[0].get_output()

        # set the normalization factor of the end effector nodes
        self._gripper_x.set_normalization_factor(int_field_output.sum())

        # compute ramps for x and y direction
        field_length_x = len(int_field_output)

        min_x = 0
        max_x = 1.0

        ramp_x = numpy.linspace(min_x, max_x , field_length_x)

        # get the force values for x towards the peak.
        gripper_x_boost = numpy.dot(int_field_output, ramp_x)

        #print("boost x: ", str(gripper_x_boost))

        current_x = self._motion_proxy.getAngles("LHand", True)[0]
        self._gripper_x.set_boost(gripper_x_boost)

        x_dot = self._gripper_x.get_change(current_x)[0]

        #print("current left hand angle: ", str(current_x))
        #print("x dot: ", str(x_dot))

        # move the hand towards the peak
        self._motion_proxy.changeAngles("LHand", x_dot, self._gripper_speed_fraction)
