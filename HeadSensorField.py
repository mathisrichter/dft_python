from naoqi import ALProxy
import numpy
import math
import DynamicField
import math_tools

class NaoHeadSensorField(DynamicField.DynamicField):
    "Nao head sensor"

    def __init__(self, camera_id = "CameraBottom", use_robot_sensors=False):
        "Constructor"
        DynamicField.DynamicField.__init__(self, dimension_bounds = [[40],[40]])

        # 2: nao space (origin in feet, x is front, y is left, z is up)
        self._robot_space_id = 2
        self._camera_id = camera_id
        self._motion_proxy = ALProxy("ALMotion", "192.168.0.102", 9559)
        self._name = "nao_head_sensor"
        self._use_robot_sensors = use_robot_sensors
        self._min_x = 0.0
        self._max_x = 0.0
        self._min_y = 0.0
        self._max_y = 0.0

        # get the orientation of the top (0 deg) or bottom camera (40 deg)

    def __del__(self):
        self._gvm_name = self._motion_proxy.unsubscribe(self._gvm_name)

    def _tilt_to_y(self, tilt, camera_height):
        return camera_height / math.tan(tilt)

    def _pan_to_x(self, pan, y):
        return y * math.sin(pan)

    def get_min_x(self):
        return self._min_x

    def get_max_x(self):
        return self._max_x

    def get_min_y(self):
        return self._min_y

    def get_max_y(self):
        return self._max_y

    def _step_computation(self):
        # get the current pan and tilt angles of the camera
        camera_pos = self._motion_proxy.getPosition(self._camera_id, self._robot_space_id, self._use_robot_sensors)
        current_head_pan = camera_pos[5]
        current_head_tilt = camera_pos[4]

        # get the current height of the objects in torso space
        # the objects are at a height of 0.35 m
        current_camera_height = camera_pos[2] - 0.35

        # compute the x,y coordinates of where the end effector should go (in
        # torso space)
#        min_tilt_angle = 0.5149 # 29.5 degrees
        min_tilt_angle = 1.2130 # 29.5 + 40.0 degrees
        max_tilt_angle = 0.1745 # 10 degrees
        self._min_y = self._tilt_to_y(min_tilt_angle, current_camera_height)
        self._max_y = self._tilt_to_y(max_tilt_angle, current_camera_height)
        current_y = self._tilt_to_y(current_head_tilt, current_camera_height)

        min_pan_angle = math.pi / 4.0
        max_pan_angle = -math.pi / 4.0
        self._min_x = self._pan_to_x(min_pan_angle, current_y)
        self._max_x = self._pan_to_x(max_pan_angle, current_y)
        current_x = self._pan_to_x(current_head_pan, current_y)

        length_y = self._max_y - self._min_y
        length_x = self._max_x - self._min_x

        # convert the target coordinates into field coordinates
        end_effector_target_y = ((current_y - self._min_y) / length_y) * self._output_dimension_sizes[1]
        end_effector_target_x = ((current_x - self._min_x) / length_x) * self._output_dimension_sizes[0]

        # create a Gaussian activation pattern at the target location
        activation = math_tools.gauss_2d(self._output_dimension_sizes, 9.0, [0.5, 0.5], [end_effector_target_x, end_effector_target_y])
        self._activation = activation

        # compute the thresholded activation of the field
        self._output_buffer = self.compute_thresholded_activation(self._activation)
