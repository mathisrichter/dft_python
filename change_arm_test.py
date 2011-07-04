import HeadSensorField
import time
import math
import numpy
from naoqi import ALProxy
import DynamicField

proxy = ALProxy("ALMotion", "nao.ini.rub.de", 9559)
proxy.setStiffnesses("RArm", 1.0)
speed_fraction = 0.5

change_sum = 1.0


#intended_pos = numpy.array([0.09, -0.11, 0.38, math.pi/2.0, 0.0, 0.0])
intended_pos = numpy.array([0.15, -0.02, 0.37, math.pi/2.0, 0.0, 0.0])

end_effector_x = DynamicField.DynamicField([], [], None)
end_effector_x.set_resting_level(0.)
end_effector_x.set_noise_strength(0.0)
end_effector_x.set_relaxation_time(50.)
end_effector_x.set_boost(intended_pos[0])

end_effector_y = DynamicField.DynamicField([], [], None)
end_effector_y.set_resting_level(0.)
end_effector_y.set_noise_strength(0.0)
end_effector_y.set_relaxation_time(50.)
end_effector_y.set_boost(intended_pos[1])

end_effector_z = DynamicField.DynamicField([], [], None)
end_effector_z.set_resting_level(0.)
end_effector_z.set_noise_strength(0.0)
end_effector_z.set_relaxation_time(50.)
end_effector_z.set_boost(intended_pos[2])

end_effector_a = DynamicField.DynamicField([], [], None)
end_effector_a.set_resting_level(0.)
end_effector_a.set_noise_strength(0.0)
end_effector_a.set_relaxation_time(50.)
end_effector_a.set_boost(intended_pos[3])


while (change_sum > 0.0002):

    pos = proxy.getPosition("RArm", 2, True)

    print(pos)

    x_dot = end_effector_x.get_change(pos[0])[0]
    y_dot = end_effector_y.get_change(pos[1])[0]
    z_dot = end_effector_z.get_change(pos[2])[0]
    a_dot = end_effector_a.get_change(pos[3])[0]

    print(x_dot)

    change = [x_dot, y_dot, z_dot, a_dot, 0.0, 0.0]

    proxy.changePosition("RArm", 2, change, speed_fraction, 15)

    change_sum = sum(numpy.fabs(change[0:4]))
    print(change_sum)

    time.sleep(0.05)

