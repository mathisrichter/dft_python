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

    def __del__(self):
        self._gvm_name = self._motion_proxy.unsubscribe(self._gvm_name)

    def _tilt_to_x(self, tilt, camera_z, camera_x):
        return (camera_z / math.tan(tilt)) + camera_x

    def _pan_to_y(self, pan, current_x, camera_y):
        return (current_x * math.sin(pan)) + camera_y

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
        cam_x = camera_pos[0]
        cam_y = camera_pos[1]
        cam_z = camera_pos[2]
        cam_pan = camera_pos[5]
        cam_tilt = camera_pos[4]

        # get the current height of the objects in torso space
        # the objects are at a height of 0.35 m
        cam_z = cam_z - 0.345

        # compute the x,y coordinates of where the end effector should go (in
        # torso space)
#        min_tilt_angle = 0.5149 # 29.5 degrees
        min_tilt_angle = 1.2130 # 29.5 + 40.0 degrees
        max_tilt_angle = 0.1745 # 10 degrees
        self._min_x = self._tilt_to_x(min_tilt_angle, cam_x, cam_z)
        self._max_x = self._tilt_to_x(max_tilt_angle, cam_x, cam_z)
        current_x = self._tilt_to_x(cam_tilt, cam_x, cam_z)

        min_pan_angle = math.pi / 4.0
        max_pan_angle = -math.pi / 4.0
        self._min_y = self._pan_to_y(min_pan_angle, current_x, cam_y)
        self._max_y = self._pan_to_y(max_pan_angle, current_x, cam_y)
        current_y = self._pan_to_y(cam_pan, current_x, cam_y)

        length_x = self._max_x - self._min_x
        length_y = self._max_y - self._min_y

#        print("current x from cam: ", current_x)
#        print("current y from cam: ", current_y)

        # convert the target coordinates into field coordinates
        end_effector_target_x = ((current_x - self._min_x) / length_x) * self._output_dimension_sizes[0]
        end_effector_target_y = ((current_y - self._min_y) / length_y) * self._output_dimension_sizes[1]

        # create a Gaussian activation pattern at the target location
        activation = math_tools.gauss_2d(self._output_dimension_sizes, 6.0, [2.0, 2.0], [end_effector_target_y, end_effector_target_x]) - 5.0
        self._activation = activation

        # compute the thresholded activation of the field
        self._output_buffer = self.compute_thresholded_activation(self._activation)
