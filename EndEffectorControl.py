import naoqi
from naoqi import ALProxy
import numpy
import DynamicField
import math_tools
import math

class EndEffectorControl(DynamicField.Connectable):
    "End effector control"

    def __init__(self, input_dimension_sizes, head_speed_fraction = 0.2, end_effector_speed_fraction = 0.2, use_robot_sensors = False):
        "Constructor"
        DynamicField.Connectable.__init__(self)
        self._head_speed_fraction = head_speed_fraction
        self._end_effector_speed_fraction = end_effector_speed_fraction
        self._use_robot_sensors = use_robot_sensors
        self._input_dimensionality = 2
        self._input_dimension_sizes = input_dimension_sizes

        # naoqi proxy to talk to the motion module
        self._motion_proxy = ALProxy("ALMotion", "192.168.0.102", 9559)
        # set the stiffness of the head and the arm to 1.0, so they will move
        self._motion_proxy.setStiffnesses("Head", 1.0)
        self._motion_proxy.setStiffnesses("RArm", 0.0) # TODO CHANGE
        # move the end effector to an initial position to make grasping
        # easier
        # [position x, pos y, pos z, orientation alpha, ori beta, ori gamma]
        initial_end_effector_configuration = [0.05, -0.10, 0.0, 0.0, 0.0, 0.0]
#        self._motion_proxy.positionInterpolation("RArm", 0, initial_end_effector_configuration, 7, 3, True)

        # move the head to an initial position
        initial_head_configuration = [0.0, 0.0]
        self._motion_proxy.angleInterpolation("Head", initial_head_configuration, 3., True)

        # dynamical system that controls the pan of the head
        self._head_node_pan = DynamicField.DynamicField([], [], None)
        self._head_node_pan.set_resting_level(0.)
        self._head_node_pan.set_initial_activation(0.)
        self._head_node_pan.set_noise_strength(0.)
        self._head_node_pan.set_relaxation_time(30.)

        # dynamical system that controls the tilt of the head
        self._head_node_tilt = DynamicField.DynamicField([], [], None)
        self._head_node_tilt.set_resting_level(0.)
        self._head_node_tilt.set_initial_activation(0.)
        self._head_node_tilt.set_noise_strength(0.)
        self._head_node_tilt.set_relaxation_time(30.)

        # dynamical system that controls the x-coordinate of the end effector
        self._end_effector_node_x = DynamicField.DynamicField([], [], None)
        self._end_effector_node_x.set_resting_level(0.)
        self._end_effector_node_x.set_initial_activation(initial_end_effector_configuration[0])
        self._end_effector_node_x.set_noise_strength(0.)
        self._end_effector_node_x.set_relaxation_time(60.)

        # dynamical system that controls the y-coordinate of the end effector
        self._end_effector_node_y = DynamicField.DynamicField([], [], None)
        self._end_effector_node_y.set_resting_level(0.)
        self._end_effector_node_y.set_initial_activation(initial_end_effector_configuration[1])
        self._end_effector_node_y.set_noise_strength(0.)
        self._end_effector_node_y.set_relaxation_time(60.)

    def __del__(self):
        self._motion_proxy.setStiffnesses("Head", 0.0)
        self._motion_proxy.setStiffnesses("RArm", 0.0)

    def _step_computation(self):
        # extract x and y position of peak (in retinal coordinates)
        int_field_output = self.get_incoming_connectables()[0].get_output()
        int_field_output_x = int_field_output.max(1)
        int_field_output_y = int_field_output.max(0)

        self._head_node_pan.set_normalization_factor(int_field_output_x.sum())
        self._head_node_tilt.set_normalization_factor(int_field_output_y.sum())

        ramp_x = range(len(int_field_output_x))
        ramp_y = range(len(int_field_output_y))

        # get the force values for x,y towards the peak
        head_force_x = numpy.dot(int_field_output_x, ramp_x)
        head_force_y = numpy.dot(int_field_output_y, ramp_y)

        # recompute the force in radiants for a horizontal opening angle of the
        # camera of 48.4 deg
        head_force_x *= -(0.8098327729 / len(int_field_output_x))
        # recompute the force in radiants for a vertical opening angle of the
        # camera of 34.8 deg
        head_force_y *= -(0.6073745796 / len(int_field_output_y))

        self._head_node_pan.set_boost(head_force_x)
        self._head_node_tilt.set_boost(head_force_y)

        # compute the change values for pan and tilt of the head
        head_pan_change = self._head_node_pan.get_change()[0]
        head_tilt_change = self._head_node_tilt.get_change()[0]

        # move the head towards the peak
        self._motion_proxy.changeAngles("HeadYaw", head_pan_change, self._head_speed_fraction)
        self._motion_proxy.changeAngles("HeadPitch", head_tilt_change, self._head_speed_fraction)

        # step the head pan and tilt nodes
        self._head_node_pan.step()
        self._head_node_tilt.step()

        # get the current pan and tilt angles of the head
        current_head_pan = self._motion_proxy.getAngles("HeadYaw", self._use_robot_sensors)[0]
        current_head_tilt = self._motion_proxy.getAngles("HeadPitch", self._use_robot_sensors)[0]

        # get the current height of the camera in torso space
        current_camera_height = self._motion_proxy.getPosition("CameraTop", 0, self._use_robot_sensors)[2]

        # compute the x,y coordinates of where the end effector should go (in
        # torso space)
        end_effector_force_x = current_camera_height * math.tan(current_head_tilt)
        end_effector_force_y = current_camera_height * math.tan(current_head_pan)
        
        self._end_effector_node_x.set_boost(end_effector_force_x)
        self._end_effector_node_y.set_boost(end_effector_force_y)

        # compute the change values for x,y of the end_effector
        end_effector_change = [self._end_effector_node_x.get_change()[0],
                               self._end_effector_node_y.get_change()[0],
                               0., # position z
                               0., # orientation alpha
                               0., # orientation beta
                               0.] # orientation gamma

        # move the head towards the peak
        # (the last parameter is the axis mask and determines, what should be
        # controlled: 7 for position only, 56 for orientation only, and 63
        # for position and orientation
        #self._motion_proxy.changePosition("RArm", 0, end_effector_change, self._end_effector_speed_fraction, 7)
