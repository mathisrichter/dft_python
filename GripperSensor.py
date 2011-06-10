from naoqi import ALProxy
import numpy
import math
import DynamicField
import math_tools

class NaoGripperSensorRight(DynamicField.DynamicField):
    "Nao gripper sensor"

    def __init__(self, gripper_field_size, use_robot_sensors=False):
        "Constructor"
        DynamicField.DynamicField.__init__(self, dimension_bounds = [[gripper_field_size]])

        self._motion_proxy = ALProxy("ALMotion", "192.168.0.102", 9559)
        self._name = "nao_gripper_sensor_right"
        self._use_robot_sensors = use_robot_sensors

    def _step_computation(self):
        # get the current position of the gripper
        gripper_pos = self._motion_proxy.getAngles("RHand", self._use_robot_sensors)[0]

        gripper_field_pos = gripper_pos * (self._output_dimension_sizes[0])

        # create a Gaussian activation pattern at the target location
        activation = math_tools.gauss_1d(self._output_dimension_sizes[0], 6.0, 1.0, gripper_field_pos) - 5.0
        self._activation = activation

        # compute the thresholded activation of the field
        self._output_buffer = self.compute_thresholded_activation(self._activation)

class NaoGripperSensorLeft(DynamicField.DynamicField):
    "Nao gripper sensor"

    def __init__(self, gripper_field_size, use_robot_sensors=False):
        "Constructor"
        DynamicField.DynamicField.__init__(self, dimension_bounds = [[gripper_field_size]])

        self._motion_proxy = ALProxy("ALMotion", "192.168.0.102", 9559)
        self._name = "nao_gripper_sensor_left"
        self._use_robot_sensors = use_robot_sensors

    def _step_computation(self):
        # get the current position of the gripper
        gripper_pos = self._motion_proxy.getAngles("LHand", self._use_robot_sensors)[0]

        gripper_field_pos = gripper_pos * (self._output_dimension_sizes[0])

        # create a Gaussian activation pattern at the target location
        activation = math_tools.gauss_1d(self._output_dimension_sizes[0], 6.0, 1.0, gripper_field_pos) - 5.0
        self._activation = activation

        # compute the thresholded activation of the field
        self._output_buffer = self.compute_thresholded_activation(self._activation)
