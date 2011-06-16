import naoqi
from naoqi import ALProxy
import numpy
import DynamicField
import math_tools
import math

class HeightOrientationRight(DynamicField.Connectable):
    "Control of height and orientation of the right end effector"

    def __init__(self, end_effector_speed_fraction = 0.5, use_robot_sensors = True):
        "Constructor"
        DynamicField.Connectable.__init__(self)
        self._end_effector_speed_fraction = end_effector_speed_fraction
        self._use_robot_sensors = use_robot_sensors

        # naoqi proxy to talk to the motion module
        self._motion_proxy = ALProxy("ALMotion", "192.168.0.102", 9559)
        # set the stiffness of the arm to 1.0, so it will move
        self._motion_proxy.setStiffnesses("RArm", 1.0)

        self._target_z = 0.35
        self._target_alpha = math.pi / 2.0

        current_pos = self._motion_proxy.getPosition("RArm", 2, True)

        self._intention_node = DynamicField.DynamicField([], [], None)
        self._intention_node.set_boost(6.0)

    def __del__(self):
        self._motion_proxy.setStiffnesses("RArm", 0.0)

    def get_intention_node(self):
        return self._intention_node

    def _get_change(self, current_value, attractor_value):
        return self._intention_node.get_output()[0] * (0.5 * (-1 * current_value + attractor_value))

    def _step_computation(self):
        self._intention_node.step()
#        print("intention node: ", self._intention_node.get_activation())
        current_pos = self._motion_proxy.getPosition("RArm", 2, self._use_robot_sensors)
        current_z = current_pos[2]
        current_alpha = current_pos[3]

#        print("current z: ", current_z)
#        print("current alpha: ", current_z)
        z_dot = self._get_change(current_z, self._target_z)
        alpha_dot = self._get_change(current_alpha, self._target_alpha)
#        print("z dot: ", z_dot)
#        print("alpha dot: ", alpha_dot)

        end_effector_change = [0.0,
                               0.0,
                               z_dot,
                               alpha_dot,
                               0., # orientation beta
                               0.] # orientation gamma

#        print("ee change: ", end_effector_change)

        self._motion_proxy.changePosition("RArm", 2, end_effector_change, self._end_effector_speed_fraction, 12)

class HeightOrientationLeft(DynamicField.Connectable):
    "Control of height and orientation of the left end effector"

    def __init__(self, end_effector_speed_fraction = 0.5, use_robot_sensors = True):
        "Constructor"
        DynamicField.Connectable.__init__(self)
        self._end_effector_speed_fraction = end_effector_speed_fraction
        self._use_robot_sensors = use_robot_sensors

        # naoqi proxy to talk to the motion module
        self._motion_proxy = ALProxy("ALMotion", "192.168.0.102", 9559)
        # set the stiffness of the arm to 1.0, so it will move
        self._motion_proxy.setStiffnesses("LArm", 1.0)

        self._target_z = 0.35
        self._target_alpha = math.pi / 2.0

        current_pos = self._motion_proxy.getPosition("LArm", 2, True)

        self._intention_node = DynamicField.DynamicField([], [], None)
        self._intention_node.set_boost(6.0)

    def __del__(self):
        self._motion_proxy.setStiffnesses("LArm", 0.0) 

    def get_intention_node(self):
        return self._intention_node

    def _get_change(self, current_value, attractor_value):
        return self._intention_node.get_output()[0] * (0.1 * (-1 * current_value + attractor_value))

    def _step_computation(self):
        self._intention_node.step()
        print("intention node: ", self._intention_node.get_activation())
        current_pos = self._motion_proxy.getPosition("LArm", 2, self._use_robot_sensors)
        current_z = current_pos[2]
        current_alpha = current_pos[3]

        print("current z: ", current_z)
        print("current alpha: ", current_z)
        z_dot = self._get_change(current_z, self._target_z)
        alpha_dot = self._get_change(current_alpha, self._target_alpha)
        print("z dot: ", z_dot)
        print("alpha dot: ", alpha_dot)

        end_effector_change = [0.0,
                               0.0,
                               z_dot,
                               alpha_dot,
                               0., # orientation beta
                               0.] # orientation gamma

#        print("ee change: ", end_effector_change)

        self._motion_proxy.changePosition("LArm", 2, end_effector_change, self._end_effector_speed_fraction, 12)


class PlaneVisualRight(DynamicField.Connectable):
    "Visual servoing of the right end effector"

    def __init__(self, input_dimension_sizes, end_effector_speed_fraction = 0.2, use_robot_sensors = True):
        "Constructor"
        DynamicField.Connectable.__init__(self)
        self._end_effector_speed_fraction = end_effector_speed_fraction
        self._use_robot_sensors = use_robot_sensors
        self._input_dimensionality = 2
        self._input_dimension_sizes = input_dimension_sizes

        # naoqi proxy to talk to the motion module
        self._motion_proxy = ALProxy("ALMotion", "192.168.0.102", 9559)
        # set the stiffness of the arm to 1.0, so it will move
        self._motion_proxy.setStiffnesses("RArm", 1.0)

        current_pos = self._motion_proxy.getPosition("RArm", 2, True)

        # create nodes controlling the end effector
        self._end_effector_x = DynamicField.DynamicField([], [], None)
        self._end_effector_x.set_resting_level(0.)
        self._end_effector_x.set_noise_strength(0.0)
        self._end_effector_x.set_relaxation_time(50.)
        self._end_effector_x.set_boost(current_pos[0])

        self._end_effector_y = DynamicField.DynamicField([], [], None)
        self._end_effector_y.set_resting_level(0.)
        self._end_effector_y.set_noise_strength(0.0)
        self._end_effector_y.set_relaxation_time(50.)
        self._end_effector_y.set_boost(current_pos[1])


    def __del__(self):
        self._motion_proxy.setStiffnesses("RArm", 0.0)
        self._file.close()

    def _step_computation(self):
        int_field_output = self.get_incoming_connectables()[0].get_output()
        int_field_output_x = int_field_output.max(0)
        int_field_output_y = int_field_output.max(1)

        field_length_x = len(int_field_output_x)
        field_length_y = len(int_field_output_y)

        print("length x: ", field_length_x)
        print("length y: ", field_length_y)

        ramp_x = numpy.linspace(-1.0, 1.0, field_length_x)
        ramp_y = numpy.linspace(-1.0, 1.0, field_length_y)

        print("x output: ", int_field_output_x)
        print("y output: ", int_field_output_y)

        x_dot = numpy.dot(int_field_output_x, ramp_x) / -1000.0
        y_dot = numpy.dot(int_field_output_y, ramp_y) /  1000.0

        print("x dot: ", x_dot)
        print("y dot: ", y_dot)

        end_effector_change = [x_dot,
                               y_dot,
                               0.0,
                               0.0,
                               0.0, # orientation beta
                               0.0] # orientation gamma

        self._motion_proxy.changePosition("RArm", 2, end_effector_change, self._end_effector_speed_fraction, 3)





class PlaneRight(DynamicField.Connectable):
    "End effector control"

    def __init__(self, head_sensor_field, input_dimension_sizes, end_effector_speed_fraction = 0.2, use_robot_sensors = True):
        "Constructor"
        DynamicField.Connectable.__init__(self)
        self._end_effector_speed_fraction = end_effector_speed_fraction
        self._use_robot_sensors = use_robot_sensors
        self._input_dimensionality = 2
        self._input_dimension_sizes = input_dimension_sizes
        self._head_sensor_field = head_sensor_field

        self._file = open("right_ee_pos_0.dat", 'w')

        # naoqi proxy to talk to the motion module
        self._motion_proxy = ALProxy("ALMotion", "192.168.0.102", 9559)
        # set the stiffness of the arm to 1.0, so it will move
        self._motion_proxy.setStiffnesses("RArm", 1.0)

        current_pos = self._motion_proxy.getPosition("RArm", 2, True)

        # create nodes controlling the end effector
        self._end_effector_x = DynamicField.DynamicField([], [], None)
        self._end_effector_x.set_resting_level(0.)
        self._end_effector_x.set_noise_strength(0.0)
        self._end_effector_x.set_relaxation_time(50.)
        self._end_effector_x.set_boost(current_pos[0])

        self._end_effector_y = DynamicField.DynamicField([], [], None)
        self._end_effector_y.set_resting_level(0.)
        self._end_effector_y.set_noise_strength(0.0)
        self._end_effector_y.set_relaxation_time(50.)
        self._end_effector_y.set_boost(current_pos[1])


    def __del__(self):
        self._motion_proxy.setStiffnesses("RArm", 0.0)
        self._file.close()

    def _step_computation(self):
        # extract x and y position of peak
        # the x and y coordinates are switched in the field, in relation to the
        # robot coordinate system
        int_field_output = self.get_incoming_connectables()[0].get_output()
        int_field_output_x = int_field_output.max(0)
        int_field_output_y = int_field_output.max(1)
#        print("int field output: ", str(int_field_output))

#        height_int_field_output = self.get_incoming_connectables()[1].get_output()
#        self._end_effector_z.set_normalization_factor(height_int_field_output.sum())
#        field_length_z = len(height_int_field_output)

#        height_ramp = numpy.linspace(0.0, 


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

#        print("ramp x: ", str(ramp_x))
#        print("ramp y: ", str(ramp_y))

        # get the force values for x,y towards the peak.
        end_effector_x_boost = numpy.dot(int_field_output_x, ramp_x)
        end_effector_y_boost = numpy.dot(int_field_output_y, ramp_y)

#        print("boost x: ", str(end_effector_x_boost))
#        print("boost y: ", str(end_effector_y_boost))

        current_pos = self._motion_proxy.getPosition("RArm", 2, True)
        current_x = current_pos[0]
        current_y = current_pos[1]
        self._end_effector_x.set_boost(end_effector_x_boost)
        self._end_effector_y.set_boost(end_effector_y_boost)

        self._file.write("" + str(current_y) + "\t" + str(current_x) + "\n")

        x_dot = self._end_effector_x.get_change(current_x)[0]
        y_dot = self._end_effector_y.get_change(current_y)[0]

#        if (x_dot is None):
#            print("x dot is none")
#        if (y_dot is None):
#            print("y dot is none")
#        print("current ee x: ", str(current_x))
#        print("current ee y: ", str(current_y))
#        print("current ee z: ", str(current_z))

#        print("x dot: ", str(x_dot))
#        print("y dot: ", str(y_dot))
#        print("z dot: ", str(z_dot))

        # compute the change values for x,y of the end_effector
        end_effector_change = [x_dot,
                               y_dot,
                               0.0,
                               0.0,
                               0., # orientation beta
                               0.] # orientation gamma

#        print("ee change: ", end_effector_change)

        # move the arm towards the peak
        # (the last parameter is the axis mask and determines, what should be
        # controlled: 7 for position only, 56 for orientation only, and 63
        # for position and orientation
        self._motion_proxy.changePosition("RArm", 2, end_effector_change, self._end_effector_speed_fraction, 3)


class PlaneLeft(DynamicField.Connectable):
    "End effector control"

    def __init__(self, head_sensor_field, input_dimension_sizes, end_effector_speed_fraction = 0.2, use_robot_sensors = True):
        "Constructor"
        DynamicField.Connectable.__init__(self)
        self._end_effector_speed_fraction = end_effector_speed_fraction
        self._use_robot_sensors = use_robot_sensors
        self._input_dimensionality = 2
        self._input_dimension_sizes = input_dimension_sizes
        self._head_sensor_field = head_sensor_field

        self._file = open("left_ee_pos_0.dat", 'w')

        # naoqi proxy to talk to the motion module
        self._motion_proxy = ALProxy("ALMotion", "192.168.0.102", 9559)
        # set the stiffness of the arm to 1.0, so it will move
        self._motion_proxy.setStiffnesses("LArm", 1.0)

        current_pos = self._motion_proxy.getPosition("LArm", 2, True)

        # create nodes controlling the end effector
        self._end_effector_x = DynamicField.DynamicField([], [], None)
        self._end_effector_x.set_resting_level(0.)
        self._end_effector_x.set_noise_strength(0.0)
        self._end_effector_x.set_relaxation_time(50.)
        self._end_effector_x.set_boost(current_pos[0])

        self._end_effector_y = DynamicField.DynamicField([], [], None)
        self._end_effector_y.set_resting_level(0.)
        self._end_effector_y.set_noise_strength(0.0)
        self._end_effector_y.set_relaxation_time(50.)
        self._end_effector_y.set_boost(current_pos[1])


    def __del__(self):
        self._motion_proxy.setStiffnesses("LArm", 0.0)
        self._file.close()

    def _step_computation(self):
        # extract x and y position of peak
        # the x and y coordinates are switched in the field, in relation to the
        # robot coordinate system
        int_field_output = self.get_incoming_connectables()[0].get_output()
        int_field_output_x = int_field_output.max(0)
        int_field_output_y = int_field_output.max(1)
#        print("int field output: ", str(int_field_output))

#        height_int_field_output = self.get_incoming_connectables()[1].get_output()
#        self._end_effector_z.set_normalization_factor(height_int_field_output.sum())
#        field_length_z = len(height_int_field_output)

#        height_ramp = numpy.linspace(0.0, 


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

#        print("ramp x: ", str(ramp_x))
#        print("ramp y: ", str(ramp_y))

        # get the force values for x,y towards the peak.
        end_effector_x_boost = numpy.dot(int_field_output_x, ramp_x)
        end_effector_y_boost = numpy.dot(int_field_output_y, ramp_y)

#        print("boost x: ", str(end_effector_x_boost))
#        print("boost y: ", str(end_effector_y_boost))

        current_pos = self._motion_proxy.getPosition("LArm", 2, True)
        current_x = current_pos[0]
        current_y = current_pos[1]
        self._end_effector_x.set_boost(end_effector_x_boost)
        self._end_effector_y.set_boost(end_effector_y_boost)

        self._file.write("" + str(current_y) + "\t" + str(current_x) + "\n")

        x_dot = self._end_effector_x.get_change(current_x)[0]
        y_dot = self._end_effector_y.get_change(current_y)[0]

#        if (x_dot is None):
#            print("x dot is none")
#        if (y_dot is None):
#            print("y dot is none")
#        print("current ee x: ", str(current_x))
#        print("current ee y: ", str(current_y))
#        print("current ee z: ", str(current_z))

#        print("x dot: ", str(x_dot))
#        print("y dot: ", str(y_dot))
#        print("z dot: ", str(z_dot))

        # compute the change values for x,y of the end_effector
        end_effector_change = [x_dot,
                               y_dot,
                               0.0,
                               0.0,
                               0., # orientation beta
                               0.] # orientation gamma

#        print("ee change: ", end_effector_change)

        # move the arm towards the peak
        # (the last parameter is the axis mask and determines, what should be
        # controlled: 7 for position only, 56 for orientation only, and 63
        # for position and orientation
        self._motion_proxy.changePosition("LArm", 2, end_effector_change, self._end_effector_speed_fraction, 3)



