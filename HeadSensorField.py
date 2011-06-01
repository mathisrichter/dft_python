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

    def _tilt_to_y(self, tilt, camera_height):
        return camera_height / math.tan(tilt)

    def _pan_to_x(self, pan, y):
        return y * math.tan(pan)

    def _step_computation(self):
        # get the current pan and tilt angles of the head
        current_head_pan = self._motion_proxy.getAngles("HeadYaw", self._use_robot_sensors)[0]
        current_head_tilt = math.fabs(self._motion_proxy.getAngles("HeadPitch", self._use_robot_sensors)[0])

        # get the current height of the camera in torso space
        current_camera_height = self._motion_proxy.getPosition("CameraTop", 0, self._use_robot_sensors)[2]

        # compute the x,y coordinates of where the end effector should go (in
        # torso space)
        min_tilt_angle = 0.5149 # 29.5 degrees
        max_tilt_angle = 0.1745 # 10 degrees
        min_y = self._tilt_to_y(min_tilt_angle, current_camera_height)
        max_y = self._tilt_to_y(max_tilt_angle, current_camera_height)
        current_y = self._tilt_to_y(current_head_tilt, current_camera_height)

        min_pan_angle = math.pi / 4.0
        max_pan_angle = -math.pi / 4.0
        min_x = self._pan_to_x(min_pan_angle, current_y)
        max_x = self._pan_to_x(max_pan_angle, current_y)
        current_x = self._pan_to_x(current_head_pan, current_y)

        length_y = max_y - min_y
        length_x = max_x - min_x

        # convert the target coordinates into field coordinates
        end_effector_target_y = ((current_y - min_y) / length_y) * self._output_dimension_sizes[1]
        end_effector_target_x = ((current_x - min_x) / length_x) * self._output_dimension_sizes[0]

        # create a Gaussian activation pattern at the target location
        activation = math_tools.gauss_2d(self._output_dimension_sizes, 9.0, [0.5, 0.5], [end_effector_target_x, end_effector_target_y])
        self._activation = activation

        # compute the thresholded activation of the field
        self._output_buffer = self.compute_thresholded_activation(self._activation)
