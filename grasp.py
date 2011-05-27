import BehavioralOrganization as BehOrg
import DynamicField
import Kernel
import numpy
import math
import matplotlib.pyplot as plt
from matplotlib import rc
from enthought.mayavi import mlab
from mpl_toolkits.axes_grid import ImageGrid
from mpl_toolkits.axes_grid import make_axes_locatable
import plot_settings
import math_tools
import CameraField
import EndEffectorControl



def main():
    grasp_architecture = BehOrg.GraspArchitecture()

    time_steps = 300

    task_node_activation = [0] * time_steps

    find_color_intention_node_activation = [0] * time_steps
    find_color_cos_node_activation = [0] * time_steps
    find_color_cos_memory_node_activation = [0] * time_steps
    find_color_intention_field_activation = numpy.zeros((time_steps, grasp_architecture._find_color_field_size))
    find_color_cos_field_activation = numpy.zeros((time_steps, grasp_architecture._find_color_field_size))

    move_ee_intention_node_activation = [0] * time_steps
    move_ee_cos_node_activation = [0] * time_steps
    move_ee_cos_memory_node_activation = [0] * time_steps
    move_ee_intention_field_activation = numpy.zeros((time_steps, grasp_architecture._move_ee_field_sizes[0]))
    move_ee_cos_field_activation = numpy.zeros((time_steps, grasp_architecture._move_ee_field_sizes[0]))

    gripper_open_intention_node_activation = [0] * time_steps
    gripper_open_cos_node_activation = [0] * time_steps
    gripper_open_cos_memory_node_activation = [0] * time_steps

    gripper_close_intention_node_activation = [0] * time_steps
    gripper_close_cos_node_activation = [0] * time_steps
    gripper_close_cos_memory_node_activation = [0] * time_steps

    gripper_intention_field_activation = numpy.zeros((time_steps, grasp_architecture._gripper_field_size))
    gripper_cos_field_activation = numpy.zeros((time_steps, grasp_architecture._gripper_field_size))

    gripper_open_precondition_node_activation = [0] * time_steps
    gripper_close_precondition_node_activation = [0] * time_steps

    color_space_field_x_activation = numpy.zeros((time_steps, grasp_architecture._color_space_field_sizes[0]))
    color_space_field_y_activation = numpy.zeros((time_steps, grasp_architecture._color_space_field_sizes[1]))
    color_space_field_hue_activation = numpy.zeros((time_steps, grasp_architecture._color_space_field_sizes[2]))

    camera_field_x_activation = numpy.zeros((time_steps, grasp_architecture._camera_field_sizes[0]))
    camera_field_y_activation = numpy.zeros((time_steps, grasp_architecture._camera_field_sizes[1]))
    camera_field_hue_activation = numpy.zeros((time_steps, grasp_architecture._camera_field_sizes[2]))

    spatial_target_field_x_activation = numpy.zeros((time_steps, grasp_architecture._spatial_target_field_sizes[0]))
    spatial_target_field_y_activation = numpy.zeros((time_steps, grasp_architecture._spatial_target_field_sizes[1]))

    perception_ee_field_x_activation = numpy.zeros((time_steps, grasp_architecture._perception_ee_field_sizes[0]))



    gripper_boost = math_tools.gauss_1d(grasp_architecture._gripper_field_size, amplitude=10.0, sigma=0.5, shift=grasp_architecture._gripper_field_size-5) 
    grasp_architecture._gripper_open.get_cos_field().set_boost(gripper_boost)

#    perception_ee_boost = math_tools.gauss_2d(perception_ee_field_sizes, amplitude=8.0, sigmas=[2.0, 2.0], shifts=[20,5])
#    perception_ee_field.set_boost(perception_ee_boost)


    for i in range(time_steps):
        print "time step: ", str(i)

        if (i == 400):
            perception_ee_boost = math_tools.gauss_2d(grasp_architecture._perception_ee_field_sizes, amplitude=8.0, sigmas=[0.5, 0.5], shifts=[5,25])
            grasp_architecture._perception_ee_field.set_boost(perception_ee_boost)


        # step all connectables and behaviors
        grasp_architecture.step()


        # save task node activation
        task_node_activation[i] = grasp_architecture._task_node.get_activation()[0]

        # save find color activations
        find_color_intention_node_activation[i] = grasp_architecture._find_color.get_intention_node().get_activation()[0]
        find_color_cos_node_activation[i] = grasp_architecture._find_color.get_cos_node().get_activation()[0]
        find_color_cos_memory_node_activation[i] = grasp_architecture._find_color.get_cos_memory_node().get_activation()[0]
        find_color_intention_field_activation[i] = grasp_architecture._find_color.get_intention_field().get_activation()
        find_color_cos_field_activation[i] = grasp_architecture._find_color.get_cos_field().get_activation()

        # save move end effector activations
        move_ee_intention_node_activation[i] = grasp_architecture._move_ee.get_intention_node().get_activation()[0]
        move_ee_cos_node_activation[i] = grasp_architecture._move_ee.get_cos_node().get_activation()[0]
        move_ee_cos_memory_node_activation[i] = grasp_architecture._move_ee.get_cos_memory_node().get_activation()[0]
        move_ee_intention_field_activation[i] = grasp_architecture._move_ee.get_intention_field().get_activation().max(1)
        move_ee_cos_field_activation[i] = grasp_architecture._move_ee.get_cos_field().get_activation().max(1)

        # save gripper open activations
        gripper_open_intention_node_activation[i] = grasp_architecture._gripper_open.get_intention_node().get_activation()[0]
        gripper_open_cos_node_activation[i] = grasp_architecture._gripper_open.get_cos_node().get_activation()[0]
        gripper_open_cos_memory_node_activation[i] = grasp_architecture._gripper_open.get_cos_memory_node().get_activation()[0]

        # save gripper close activations
        gripper_close_intention_node_activation[i] = grasp_architecture._gripper_close.get_intention_node().get_activation()[0]
        gripper_close_cos_node_activation[i] = grasp_architecture._gripper_close.get_cos_node().get_activation()[0]
        gripper_close_cos_memory_node_activation[i] = grasp_architecture._gripper_close.get_cos_memory_node().get_activation()[0]

        # save gripper field activations
        gripper_intention_field_activation[i] = grasp_architecture._gripper_intention_field.get_activation()
        gripper_cos_field_activation[i] = grasp_architecture._gripper_cos_field.get_activation()

        # save precondition activations
        gripper_open_precondition_node_activation[i] = grasp_architecture._gripper_open_precondition_node.get_activation()[0]
        gripper_close_precondition_node_activation[i] = grasp_architecture._gripper_close_precondition_node.get_activation()[0]

        # save color space field activations
        color_space_field_hue_x_activation = grasp_architecture._color_space_field.get_activation().max(1)
        color_space_field_hue_activation[i] = color_space_field_hue_x_activation.max(0)
        color_space_field_x_activation[i] = color_space_field_hue_x_activation.max(1)
        color_space_field_y_activation[i] = grasp_architecture._color_space_field.get_activation().max(0).max(1)

        # save camera field activations
        camera_field_hue_x_activation = grasp_architecture._camera_field.get_activation().max(1)
        camera_field_hue_activation[i] = camera_field_hue_x_activation.max(0)
        camera_field_x_activation[i] = camera_field_hue_x_activation.max(1)
        camera_field_y_activation[i] = grasp_architecture._camera_field.get_activation().max(0).max(1)

        # save spatial target activations
        spatial_target_field_x_activation[i] = grasp_architecture._spatial_target_field.get_activation().max(1)
        spatial_target_field_y_activation[i] = grasp_architecture._spatial_target_field.get_activation().max(0)

        # save perception end effector activations
        perception_ee_field_x_activation[i] = grasp_architecture._perception_ee_field.get_activation().max(1)

    plot_settings.set_mode("icdl")

    # create a figure for the "find color" plots
    fig = plt.figure(1)
    fig.subplots_adjust(bottom=0.07, left=0.07, right=0.97, top=0.93)

    plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])

    time_course_subplot = plt.subplot(2,1,1)
    time_course_subplot.axes.grid(color='grey', linestyle='dotted')

    plt.xlabel(r'time steps')
    plt.ylabel(r'activation')
    
    plt.plot(task_node_activation, 'y-', label=r'task')

    plt.plot(find_color_intention_node_activation, 'r-', label=r'fc intention', antialiased=True)
    plt.plot(find_color_cos_node_activation, 'b-', label=r'fc cos', antialiased=True)
    plt.plot(find_color_cos_memory_node_activation, 'c-', label=r'fc cos mem', antialiased=True)

    plt.plot(gripper_open_precondition_node_activation, 'g-.', label=r'open precondition', antialiased=True)
    plt.plot(gripper_close_precondition_node_activation, 'm-.', label=r'close precondition', antialiased=True)
    plt.legend(loc='upper right')

    grid = ImageGrid(fig, 212, nrows_ncols = (2,1), axes_pad=0.1, aspect=False)

    grid[0].imshow(numpy.rollaxis(find_color_intention_field_activation, 1), aspect="auto", vmin=-10, vmax=10)
    grid[0].invert_yaxis()
    grid[0].set_yticks(range(0,grasp_architecture._find_color_field_size+10,20))
    grid[0].set_ylabel(r'fc int')

    grid[1].imshow(numpy.rollaxis(find_color_cos_field_activation, 1), aspect="auto", vmin=-10, vmax=10)
    grid[1].invert_yaxis()
    grid[1].set_yticks(range(0,grasp_architecture._find_color_field_size+10,20))
    grid[1].set_ylabel(r'fc cos')
    grid[1].set_xlabel(r'time steps')
    grid[1].set_xticks(range(0,time_steps+100,200))

    plt.savefig("fig/find_color.pdf", format="pdf")

    ##########################################################################

    # create a figure for the "move ee" plots
    fig = plt.figure(2)
    fig.subplots_adjust(bottom=0.07, left=0.07, right=0.97, top=0.93)

    plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])

    time_course_subplot = plt.subplot(2,1,1)
    time_course_subplot.axes.grid(color='grey', linestyle='dotted')

    plt.xlabel(r'time steps')
    plt.ylabel(r'activation')
    
    plt.plot(move_ee_intention_node_activation, 'r-', label=r'mee intention', antialiased=True)
    plt.plot(move_ee_cos_node_activation, 'b-', label=r'mee cos', antialiased=True)
    plt.plot(move_ee_cos_memory_node_activation, 'c-', label=r'mee cos mem', antialiased=True)

    plt.legend(loc='upper right')

    grid = ImageGrid(fig, 212, nrows_ncols = (5,1), axes_pad=0.1, aspect=False)

    grid[0].imshow(numpy.rollaxis(move_ee_intention_field_activation, 1), aspect="auto", vmin=-10, vmax=10)
    grid[0].invert_yaxis()
    grid[0].set_yticks(range(0,grasp_architecture._move_ee_field_sizes[0]+10,20))
    grid[0].set_ylabel(r'mee int')

    grid[1].imshow(numpy.rollaxis(move_ee_cos_field_activation, 1), aspect="auto", vmin=-10, vmax=10)
    grid[1].invert_yaxis()
    grid[1].set_yticks(range(0,grasp_architecture._move_ee_field_sizes[0]+10,20))
    grid[1].set_ylabel(r'mee cos')

    grid[2].imshow(numpy.rollaxis(spatial_target_field_x_activation, 1), aspect="auto", vmin=-10, vmax=10)
    grid[2].invert_yaxis()
    grid[2].set_yticks(range(0,grasp_architecture._spatial_target_field_sizes[0]+10,20))
    grid[2].set_ylabel(r'st x')

    grid[3].imshow(numpy.rollaxis(spatial_target_field_y_activation, 1), aspect="auto", vmin=-10, vmax=10)
    grid[3].invert_yaxis()
    grid[3].set_yticks(range(0,grasp_architecture._spatial_target_field_sizes[1]+10,20))
    grid[3].set_ylabel(r'st y')

    grid[4].imshow(numpy.rollaxis(perception_ee_field_x_activation, 1), aspect="auto", vmin=-10, vmax=10)
    grid[4].invert_yaxis()
    grid[4].set_yticks(range(0,grasp_architecture._perception_ee_field_sizes[0]+10,20))
    grid[4].set_ylabel(r'pe x')

    grid[4].set_xlabel(r'time steps')
    grid[4].set_xticks(range(0,time_steps+100,200))

    plt.savefig("fig/move_ee.pdf", format="pdf")

    ##########################################################################

    # create a figure for the "gripper" plots
    fig = plt.figure(3)
    fig.subplots_adjust(bottom=0.07, left=0.07, right=0.97, top=0.93)

    plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])

    time_course_subplot = plt.subplot(2,1,1)
    time_course_subplot.axes.grid(color='grey', linestyle='dotted')

    plt.xlabel(r'time steps')
    plt.ylabel(r'activation')
    
    plt.plot(gripper_open_intention_node_activation, 'r-', label=r'go intention', antialiased=True)
    plt.plot(gripper_open_cos_node_activation, 'b-', label=r'go cos', antialiased=True)
    plt.plot(gripper_open_cos_memory_node_activation, 'c-', label=r'go cos mem', antialiased=True)

    plt.plot(gripper_close_intention_node_activation, 'r--', label=r'gc intention', antialiased=True)
    plt.plot(gripper_close_cos_node_activation, 'b--', label=r'gc cos', antialiased=True)
    plt.plot(gripper_close_cos_memory_node_activation, 'c--', label=r'gc cos mem', antialiased=True)

    plt.legend(loc='upper right')

    grid = ImageGrid(fig, 212, nrows_ncols = (2,1), axes_pad=0.1, aspect=False)

    grid[0].imshow(numpy.rollaxis(gripper_intention_field_activation, 1), aspect="auto", vmin=-10, vmax=10)
    grid[0].invert_yaxis()
    grid[0].set_yticks(range(0,grasp_architecture._gripper_field_size+10,20))
    grid[0].set_ylabel(r'go int')

    grid[1].imshow(numpy.rollaxis(gripper_cos_field_activation, 1), aspect="auto", vmin=-10, vmax=10)
    grid[1].invert_yaxis()
    grid[1].set_yticks(range(0,grasp_architecture._gripper_field_size+10,20))
    grid[1].set_ylabel(r'go cos')
    grid[1].set_xlabel(r'time steps')
    grid[1].set_xticks(range(0,time_steps+100,200))

    plt.savefig("fig/gripper.pdf", format="pdf")

    ##########################################################################

    # create a figure for the color space field plots
    fig = plt.figure(4)
    fig.subplots_adjust(bottom=0.07, left=0.07, right=0.97, top=0.93)

    plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])

    time_course_subplot = plt.subplot(2,1,1)
    time_course_subplot.axes.grid(color='grey', linestyle='dotted')

    plt.plot(find_color_intention_node_activation, 'r-', label=r'fc intention', antialiased=True)

    plt.xlabel(r'time steps')
    plt.ylabel(r'activation')
    
    plt.legend(loc='upper right')

    grid = ImageGrid(fig, 212, nrows_ncols = (3,1), axes_pad=0.1, aspect=False)

    grid[0].imshow(numpy.rollaxis(color_space_field_x_activation, 1), aspect="auto", vmin=-10, vmax=10)
    grid[0].invert_yaxis()
    grid[0].set_yticks(range(0,grasp_architecture._color_space_field_sizes[0]+10,20))
    grid[0].set_ylabel(r'cs x')

    grid[1].imshow(numpy.rollaxis(color_space_field_y_activation, 1), aspect="auto", vmin=-10, vmax=10)
    grid[1].invert_yaxis()
    grid[1].set_yticks(range(0,grasp_architecture._color_space_field_sizes[1]+10,20))
    grid[1].set_ylabel(r'cs y')

    grid[2].imshow(numpy.rollaxis(color_space_field_hue_activation, 1), aspect="auto", vmin=-10, vmax=10)
    grid[2].invert_yaxis()
    grid[2].set_yticks(range(0,grasp_architecture._color_space_field_sizes[2]+10,20))
    grid[2].set_ylabel(r'cs hue')

    grid[2].set_xlabel(r'time steps')
    grid[2].set_xticks(range(0,time_steps+100,200))

    plt.savefig("fig/color_space.pdf", format="pdf")

    ##########################################################################

    # create a figure for the camera field plots
    fig = plt.figure(5)
    fig.subplots_adjust(bottom=0.07, left=0.07, right=0.97, top=0.93)

    plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])

    time_course_subplot = plt.subplot(2,1,1)
    time_course_subplot.axes.grid(color='grey', linestyle='dotted')

    plt.plot(find_color_intention_node_activation, 'r-', label=r'fc intention', antialiased=True)

    plt.xlabel(r'time steps')
    plt.ylabel(r'activation')
    
    plt.legend(loc='upper right')

    grid = ImageGrid(fig, 212, nrows_ncols = (3,1), axes_pad=0.1, aspect=False)

    grid[0].imshow(numpy.rollaxis(camera_field_x_activation, 1), aspect="auto", vmin=-10, vmax=10)
    grid[0].invert_yaxis()
    grid[0].set_yticks(range(0,grasp_architecture._camera_field_sizes[0]+10,20))
    grid[0].set_ylabel(r'cam x')

    grid[1].imshow(numpy.rollaxis(camera_field_y_activation, 1), aspect="auto", vmin=-10, vmax=10)
    grid[1].invert_yaxis()
    grid[1].set_yticks(range(0,grasp_architecture._camera_field_sizes[1]+10,20))
    grid[1].set_ylabel(r'cam y')

    grid[2].imshow(numpy.rollaxis(camera_field_hue_activation, 1), aspect="auto", vmin=-10, vmax=10)
    grid[2].invert_yaxis()
    grid[2].set_yticks(range(0,grasp_architecture._camera_field_sizes[2]+10,20))
    grid[2].set_ylabel(r'cam hue')

    grid[2].set_xlabel(r'time steps')
    grid[2].set_xticks(range(0,time_steps+100,200))

    plt.savefig("fig/camera.pdf", format="pdf")


    plt.show()

if __name__ == "__main__":
    main()
