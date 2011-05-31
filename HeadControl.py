import naoqi
from naoqi import ALProxy
import numpy
import DynamicField
import math_tools
import math

class HeadControl(DynamicField.Connectable):
    "Head control"

    def __init__(self, input_dimension_sizes, head_speed_fraction = 0.2, head_time_scale = 0.01, use_robot_sensors = False):
        "Constructor"
        DynamicField.Connectable.__init__(self)
        self._head_speed_fraction = head_speed_fraction
        self._use_robot_sensors = use_robot_sensors
        self._input_dimensionality = 2
        self._input_dimension_sizes = input_dimension_sizes
        self._head_time_scale = head_time_scale

        # naoqi proxy to talk to the motion module
        self._motion_proxy = ALProxy("ALMotion", "192.168.0.102", 9559)
        # set the stiffness of the head to 1.0, so it will move
        self._motion_proxy.setStiffnesses("Head", 1.0)

    def __del__(self):
        self._motion_proxy.setStiffnesses("Head", 0.0)

    def _step_computation(self):
        # extract x and y position of peak (in retinal coordinates)
        int_field_output = self.get_incoming_connectables()[0].get_output()
        int_field_output_x = int_field_output.max(1)
        int_field_output_y = int_field_output.max(0)

        # compute ramps for x and y direction
        length_x = len(int_field_output_x)
        length_y = len(int_field_output_y)

        ramp_x = numpy.linspace(-length_x/2.0, length_x/2.0, length_x)
        ramp_y = numpy.linspace(-length_y/2.0, length_y/2.0, length_y)

        # get the force values for x,y towards the peak.
        head_force_x = numpy.dot(int_field_output_x, ramp_x)
        head_force_y = numpy.dot(int_field_output_y, ramp_y)

        # opening angle of the camera in x direction (48.4 deg)
        opening_angle_x = 0.8098327729
        # opening angle of the camera in y direction (34.8 deg)
        opening_angle_y = 0.6073745796

        # compute the change values for pan and tilt of the head
        # the first factor (-1) is multiplied because the x- and y-axis of the
        # robot are reversed with respect to the field
        head_pan_change = -1 * self._head_time_scale * head_force_x * (opening_angle_x / length_x)
        head_tilt_change = -1 * self._head_time_scale * head_force_y * (opening_angle_y / length_y)

        # move the head towards the peak
        self._motion_proxy.changeAngles("HeadYaw", head_pan_change, self._head_speed_fraction)
        self._motion_proxy.changeAngles("HeadPitch", head_tilt_change, self._head_speed_fraction)
