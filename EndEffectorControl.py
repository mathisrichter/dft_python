import naoqi
from naoqi import ALProxy
import numpy
import DynamicField
import math_tools

class EndEffectorControl(DynamicField.Connectable):
    "End effector control"

    def __init__(self, head_speed_fraction = 0.2, end_effector_speed_fraction = 0.2, use_robot_sensors = False):
        "Constructor"
        self._head_speed_fraction = head_speed_fraction
        self._end_effector_speed_fraction = end_effector_speed_fraction
        self._use_robot_sensors = use_robot_sensors

        # naoqi proxy to talk to the motion module
        self._motion_proxy = ALProxy("ALMotion", "192.168.0.102", 9559)
        # set the stiffness of the head and the arm to 1.0, so they will move
        motion_proxy.setStiffnesses("Head", 1.0)
        motion_proxy.setStiffnesses("RArm", 1.0)
        # move the end effector to an initial position to make grasping
        # easier
        # [position x, pos y, pos z, orientation alpha, ori beta, ori gamma]
        initial_end_effector_configuration = [0.05, -0.10, 0.0, 0.0, 0.0, 0.0]
        motion_proxy.positionInterpolation("RArm", 0, initial_end_effector_configuration, 7, 3, True)

        # dynamical system that controls the pan of the head
        head_node_pan = DynamicField.DynamicField([], [], None)
        head_node_pan = set_resting_level(0.)
        head_node_pan.set_initial_activation(0.)
        head_node_pan.set_noise_strength(0.)
        head_node_pan.set_relaxation_time(60.)

        # dynamical system that controls the tilt of the head
        head_node_tilt = DynamicField.DynamicField([], [], None)
        head_node_tilt = set_resting_level(0.)
        head_node_tilt.set_initial_activation(0.)
        head_node_tilt.set_noise_strength(0.)
        head_node_tilt.set_relaxation_time(60.)

        # dynamical system that controls the x-coordinate of the end effector
        end_effector_node_x = DynamicField.DynamicField([], [], None)
        end_effector_node_x = set_resting_level(0.)
        end_effector_node_x.set_initial_activation(initial_end_effector_x)
        end_effector_node_x.set_noise_strength(0.)
        end_effector_node_x.set_relaxation_time(60.)

        # dynamical system that controls the y-coordinate of the end effector
        end_effector_node_y = DynamicField.DynamicField([], [], None)
        end_effector_node_y = set_resting_level(0.)
        end_effector_node_y.set_initial_activation(initial_end_effector_y)
        end_effector_node_y.set_noise_strength(0.)
        end_effector_node_y.set_relaxation_time(60.)

    def _step_computation(self):
        # extract x and y position of peak (in retinal coordinates)
        int_field_output = self.get_incoming_connectables()[0].get_output()
        int_field_output_x = int_field_output.max(1)
        int_field_output_y = int_field_output.max(0)

        head_node_pan.set_normalization_factor(int_field_output_x.sum())
        head_node_tilt.set_normalization_factor(int_field_output_y.sum())

        ramp_x = range(len(int_field_output_x)
        ramp_y = range(len(int_field_output_y)

        # get the force values for x,y towards the peak
        # TODO this is divided by 100 because of the size of the intention
        # field. change to be dependent on the resolution of the field
        head_force_x = numpy.dot(int_field_output_x, ramp_x) / 100.
        head_force_y = numpy.dot(int_field_output_y, ramp_y) / 100.

        head_node_pan.set_boost(head_force_x)
        head_node_tilt.set_boost(head_force_y)

        # compute the change values for pan and tilt of the head
        head_pan_change = head_node_pan.get_change()[0]
        head_tilt_change = head_node_tilt.get_change()[0]

        # move the head towards the peak
        self._motion_proxy.changeAngles("HeadPan", head_pan_change, self._head_speed_fraction)
        self._motion_proxy.changeAngles("HeadTilt", head_tilt_change, self._head_speed_fraction)

        # step the head pan and tilt nodes
        head_node_pan.step()
        head_node_tilt.step()

        # get the current pan and tilt angles of the head
        current_head_pan = self._motion_proxy.getAngles("HeadPan", self._use_robot_sensors)
        current_head_tilt = self._motion_proxy.getAngles("HeadTilt", self._use_robot_sensors)

        # get the current height of the camera in torso space
        current_camera_height = self._motion_proxy.getPosition("CameraTop", 0, self._use_robot_sensors)

        # compute the x,y coordinates of where the end effector should go (in
        # torso space)
        end_effector_force_x = current_camera_height * math.tan(current_head_tilt)
        end_effector_force_y = current_camera_height * math.tan(current_head_pan)
        
        end_effector_node_x.set_boost(end_effector_force_x)
        end_effector_node_y.set_boost(end_effector_force_y)

        # compute the change values for x,y of the end_effector
        end_effector_change = [end_effector_node_x.get_change()[0],
                               end_effector_node_y.get_change()[0],
                               0., # position z
                               0., # orientation alpha
                               0., # orientation beta
                               0.] # orientation gamma

        # move the head towards the peak
        # (the last parameter is the axis mask and determines, what should be
        # controlled: 7 for position only, 56 for orientation only, and 63
        # for position and orientation
        self._motion_proxy.changePosition("RArm", 0, end_effector_change, self._end_effector_speed_fraction, 7)
