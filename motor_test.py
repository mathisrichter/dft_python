import BehavioralOrganization as BehOrg
import DynamicField
import Kernel
import numpy
import math
import matplotlib.pyplot as plt
from matplotlib import rc
#from enthought.mayavi import mlab
from mpl_toolkits.axes_grid import ImageGrid
from mpl_toolkits.axes_grid import make_axes_locatable
import plot_settings
import math_tools

import naoqi
from naoqi import ALProxy
import time


def main():
    print_output = False
    move_nao = True

    motion_proxy = ALProxy("ALMotion", "192.168.0.102", 9559)
    initial_x = 0.05
    initial_y = -0.10
    initial_z = 0.0

    if (move_nao):
        motion_proxy.setStiffnesses("RArm", 1.0)
        motion_proxy.positionInterpolation("RArm", 0, [initial_x, initial_y, initial_z, 0, 0, 0], 7, 3, True)

    # create a task node
    task_node = DynamicField.DynamicField([], [], None)
    task_node.set_boost(10)

    # create a motor node
    motor_node_x = DynamicField.DynamicField([], [], None)
    motor_node_x.set_resting_level(0.)
    motor_node_x.set_initial_activation(initial_x)
    motor_node_x.set_noise_strength(0.0)
    motor_node_x.set_relaxation_time(60.)
    motor_node_y = DynamicField.DynamicField([], [], None)
    motor_node_y.set_resting_level(0.)
    motor_node_y.set_initial_activation(initial_y)
    motor_node_y.set_noise_strength(0.0)
    motor_node_y.set_relaxation_time(60.)

    # create elementary behavior: move end effector
    move_ee_field_sizes = [50, 50]
    move_ee_int_weight = numpy.ones((move_ee_field_sizes)) * 2.0
    move_ee = BehOrg.ElementaryBehavior.with_internal_fields(field_dimensionality=2,
                                                field_sizes=[[move_ee_field_sizes[0]],[move_ee_field_sizes[1]]],
                                                field_resolutions=[],
                                                int_node_to_int_field_weight=move_ee_int_weight,
                                                name="move ee")

    # connect all elementary behaviors to the task node
#    BehOrg.connect_to_task(task_node, move_ee)



    time_steps = 1000

    task_node_activation = [0] * time_steps
    motor_node_x_activation = [0] * time_steps
    motor_node_y_activation = [0] * time_steps

    move_ee_intention_node_activation = [0] * time_steps
    move_ee_cos_node_activation = [0] * time_steps
    move_ee_cos_memory_node_activation = [0] * time_steps
    move_ee_intention_field_activation = numpy.zeros((time_steps, move_ee_field_sizes[0]))
    move_ee_cos_field_activation = numpy.zeros((time_steps, move_ee_field_sizes[0]))

    for i in range(time_steps):

        # step all connectables and behaviors
        task_node.step()
        move_ee.step()

        # extract position of peak in the move ee intention field
        move_ee_int_field = move_ee.get_intention_field()
        move_ee_int_field_activation = move_ee_int_field.get_activation()

        if (i == 100):
            print "time to boost!"
            move_ee_boost = math_tools.gauss_2d(move_ee_int_field_activation.shape, amplitude=9.5, sigmas=[2.0, 2.0], shifts=[10,0])
            move_ee_int_field.set_boost(move_ee_boost)

        move_ee_int_field_activation_x = move_ee_int_field_activation.max(1)
        move_ee_int_field_activation_y = move_ee_int_field_activation.max(0)

        move_ee_int_field_output_x = move_ee_int_field.get_output().max(1)
        move_ee_int_field_output_y = move_ee_int_field.get_output().max(0)

        if (print_output):
            print "normalization x:"
            print move_ee_int_field_output_x.sum()
            print "normalization y:"
            print move_ee_int_field_output_y.sum()


        motor_node_x.set_normalization_factor(move_ee_int_field_output_x.sum())
        motor_node_y.set_normalization_factor(move_ee_int_field_output_y.sum())

        if (print_output):
            print "field x:"
            print move_ee_int_field_output_x
            print "field y:"
            print move_ee_int_field_output_y

        field_size_x = len(move_ee_int_field_output_x)
        field_size_y = len(move_ee_int_field_output_y)

#        ramp_x = numpy.linspace(-field_size_x/2., field_size_x/2., field_size_x)
#        ramp_y = numpy.linspace(-field_size_y/2., field_size_y/2., field_size_y)
        ramp_x = range(field_size_x)
        ramp_y = range(field_size_y)

        if (print_output):
            print "ramp x:"
            print ramp_x
            print "ramp y:"
            print ramp_y

        force_x = numpy.dot(move_ee_int_field_output_x, ramp_x) / 100.
        force_y = numpy.dot(move_ee_int_field_output_y, ramp_y) / -100.

        if (print_output):
            print "force x:"
            print force_x
            print "force y:"
            print force_y

        motor_node_x.set_boost(force_x)
        motor_node_y.set_boost(force_y)

        x_dot = motor_node_x.get_change()[0]
        y_dot = motor_node_y.get_change()[0]
        z_dot = 0.0

        if (print_output):
            print "change x:"
            print x_dot
            print "change y:"
            print y_dot

        motor_node_x.step()
        motor_node_y.step()

        if (move_nao):
            motion_proxy.changePosition("RArm", 0, [x_dot, y_dot, z_dot, 0.0, 0.0, 0.0], 0.5, 7)
            time.sleep(0.03)

        if (print_output):
            print "motor node x activation:"
            print motor_node_x.get_activation()[0]
            print "motor node y activation:"
            print motor_node_y.get_activation()[0]

        # save task node activation
        task_node_activation[i] = task_node.get_activation()[0]

        # save motor node activation
        motor_node_x_activation[i] = motor_node_x.get_activation()[0]
        motor_node_y_activation[i] = motor_node_y.get_activation()[0]

        # save move end effector activations
        move_ee_intention_node_activation[i] = move_ee.get_intention_node().get_activation()[0]
        move_ee_cos_node_activation[i] = move_ee.get_cos_node().get_activation()[0]
        move_ee_cos_memory_node_activation[i] = move_ee.get_cos_memory_node().get_activation()[0]
        move_ee_intention_field_activation[i] = move_ee.get_intention_field().get_activation().max(1)
        move_ee_cos_field_activation[i] = move_ee.get_cos_field().get_activation().max(1)

    if (print_output):
        print "done computing..."

    plot_settings.set_mode("icdl")

    ##########################################################################

    # create a figure for the "move ee" plots
    fig = plt.figure(1)
    fig.subplots_adjust(bottom=0.07, left=0.07, right=0.97, top=0.93)

    plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])

    time_course_subplot = plt.subplot(2,1,1)
    time_course_subplot.axes.grid(color='grey', linestyle='dotted')

    plt.xlabel(r'time steps')
    plt.ylabel(r'activation')
    
    plt.plot(task_node_activation, 'y-', label=r'task')
    plt.plot(motor_node_x_activation, 'k-', label=r'motor x')
    plt.plot(motor_node_y_activation, 'k--', label=r'motor y')

    plt.plot(move_ee_intention_node_activation, 'r-', label=r'mee intention', antialiased=True)
    plt.plot(move_ee_cos_node_activation, 'b-', label=r'mee cos', antialiased=True)
    plt.plot(move_ee_cos_memory_node_activation, 'c-', label=r'mee cos mem', antialiased=True)

    plt.axis([0,1000,-0.25,0.25])
    plt.legend(loc='upper right')

    grid = ImageGrid(fig, 212, nrows_ncols = (2,1), axes_pad=0.1, aspect=False)

    grid[0].imshow(numpy.rollaxis(move_ee_intention_field_activation, 1), aspect="auto", vmin=-10, vmax=10)
    grid[0].invert_yaxis()
    grid[0].set_yticks(range(0,move_ee_field_sizes[0]+10,20))
    grid[0].set_ylabel(r'mee int')

    grid[1].imshow(numpy.rollaxis(move_ee_cos_field_activation, 1), aspect="auto", vmin=-10, vmax=10)
    grid[1].invert_yaxis()
    grid[1].set_yticks(range(0,move_ee_field_sizes[0]+10,20))
    grid[1].set_ylabel(r'mee cos')

    grid[1].set_xlabel(r'time steps')
    grid[1].set_xticks(range(0,time_steps+100,200))

    plt.savefig("fig/move_ee.pdf", format="pdf")

    ##########################################################################

if __name__ == "__main__":
    main()
