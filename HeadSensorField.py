from naoqi import ALProxy
import numpy
import math
import DynamicField
import math_tools

class NaoHeadSensorField(DynamicField.DynamicField):
    "Nao head sensor"

    def __init__(self, use_robot_sensors=False):
        "Constructor"
        DynamicField.DynamicField.__init__(self, dimension_bounds = [[40],[40]])

        self._motion_proxy = ALProxy("ALMotion", "192.168.0.102", 9559)
        self._name = "nao_head_sensor"
        self._use_robot_sensors = use_robot_sensors

    def __del__(self):
        self._gvm_name = self._motion_proxy.unsubscribe(self._gvm_name)

    def _step_computation(self):
        # get the current pan and tilt angles of the head
        current_head_pan = self._motion_proxy.getAngles("HeadYaw", self._use_robot_sensors)[0]
        current_head_tilt = self._motion_proxy.getAngles("HeadPitch", self._use_robot_sensors)[0]

        print("pan: ", current_head_pan)
        print("tilt: ", current_head_tilt)

        # get the current height of the camera in torso space
        current_camera_height = self._motion_proxy.getPosition("CameraTop", 0, self._use_robot_sensors)[2]

        print("cam height: ", current_camera_height)

        # compute the x,y coordinates of where the end effector should go (in
        # torso space)
        end_effector_target_x = current_camera_height * math.tan(current_head_tilt)
        end_effector_target_y = current_camera_height * math.tan(current_head_pan)

        print("target x: ", end_effector_target_x)
        print("target y: ", end_effector_target_y)

        # convert the target coordinates into field coordinates
        end_effector_target_x = end_effector_target_x * self._output_dimension_sizes[0] + (self._output_dimension_sizes[0] / 2.0)
        end_effector_target_y = end_effector_target_y * self._output_dimension_sizes[1] + (self._output_dimension_sizes[1] / 2.0)

        print("targetf x: ", end_effector_target_x)
        print("targetf y: ", end_effector_target_y)

        # create a Gaussian activation pattern at the target location
        activation = math_tools.gauss_2d(self._output_dimension_sizes, 9.0, [0.5, 0.5], [end_effector_target_x, end_effector_target_y])
        self._activation = activation

        # compute the thresholded activation of the field
        self._output_buffer = self.compute_thresholded_activation(self._activation)
