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

    # create a task node
    task_node = DynamicField.DynamicField([], [], None)
    task_node.set_boost(10)

    # create elementary behavior: find color
    find_color_field_size = 15
    find_color_int_weight = math_tools.gauss_1d(find_color_field_size, amplitude=15.0, sigma=2.0, shift=0)

    find_color = BehOrg.ElementaryBehavior.with_internal_fields(field_dimensionality=1,
                                                field_sizes=[[find_color_field_size]],
                                                field_resolutions=[],
                                                int_node_to_int_field_weight=find_color_int_weight,
                                                name="find color")

    # create elementary behavior: move end effector
    move_ee_field_sizes = [40, 30]
    move_ee_int_weight = numpy.ones((move_ee_field_sizes)) * 4.0
    move_ee = BehOrg.ElementaryBehavior.with_internal_fields(field_dimensionality=2,
                                                field_sizes=[[move_ee_field_sizes[0]],[move_ee_field_sizes[1]]],
                                                field_resolutions=[],
                                                int_node_to_int_field_weight=move_ee_int_weight,
                                                name="move ee")

    # create gripper intention and cos fields
    gripper_field_dimensionality = 1
    gripper_field_size = 15

    # gripper intention field and its kernel
    intention_field_kernel = Kernel.GaussKernel(gripper_field_dimensionality)
    intention_field_kernel.add_mode(15.0, [1.0] * gripper_field_dimensionality, [0.0] * gripper_field_dimensionality)
    intention_field_kernel.calculate()
    gripper_intention_field = DynamicField.DynamicField([[gripper_field_size]], [], intention_field_kernel)
    gripper_intention_field.set_global_inhibition(200.0)

    # gripper CoS field and its kernel
    cos_field_kernel = Kernel.GaussKernel(gripper_field_dimensionality)
    cos_field_kernel.add_mode(15.0, [1.0] * gripper_field_dimensionality, [0.0] * gripper_field_dimensionality)
    cos_field_kernel.calculate()
    gripper_cos_field = DynamicField.DynamicField([[gripper_field_size]], [], cos_field_kernel)
    gripper_cos_field.set_global_inhibition(200.0)

    # connect the gripper intention and CoS field
    gripper_int_field_to_cos_field_weight = DynamicField.Weight(2.5)
    DynamicField.connect(gripper_intention_field, gripper_cos_field, [gripper_int_field_to_cos_field_weight])

    # create elementary behavior: gripper close
    gripper_close_int_weight = math_tools.gauss_1d(gripper_field_size, amplitude=15, sigma=2.0, shift=5)
    gripper_close = BehOrg.ElementaryBehavior(intention_field=gripper_intention_field,
                                              cos_field=gripper_cos_field,
                                              int_node_to_int_field_weight=gripper_close_int_weight,
                                              name="gripper close")

    # create elementary behavior: gripper open
    gripper_open_int_weight = math_tools.gauss_1d(gripper_field_size, amplitude=15, sigma=2.0, shift=gripper_field_size-5)
    gripper_open = BehOrg.ElementaryBehavior(intention_field=gripper_intention_field,
                                              cos_field=gripper_cos_field,
                                              int_node_to_int_field_weight=gripper_open_int_weight,
                                              name="gripper open")
    gripper_intention_field.set_name("gripper_intention_field")
    gripper_cos_field.set_name("gripper_cos_field")


    # connect all elementary behaviors to the task node
    BehOrg.connect_to_task(task_node, find_color)
    BehOrg.connect_to_task(task_node, move_ee)
    BehOrg.connect_to_task(task_node, gripper_open)
    BehOrg.connect_to_task(task_node, gripper_close)

    # create precondition nodes
    gripper_open_precondition_node = BehOrg.precondition(gripper_open, move_ee, task_node)
    gripper_close_precondition_node = BehOrg.precondition(move_ee, gripper_close, task_node)

    # create perception color-space field
    color_space_field_dimensionality = 3
    color_space_kernel = Kernel.GaussKernel(color_space_field_dimensionality)
    color_space_kernel.add_mode(5.0, [1.0] * color_space_field_dimensionality, [0.0] * color_space_field_dimensionality)
    color_space_kernel.calculate()

    color_space_field_sizes = [move_ee_field_sizes[0], move_ee_field_sizes[1], find_color_field_size]
    color_space_field = DynamicField.DynamicField([[color_space_field_sizes[0]],[color_space_field_sizes[1]],[color_space_field_sizes[2]]], [], color_space_kernel)
    color_space_field.set_global_inhibition(400.0)
    color_space_field.set_relaxation_time(2.0)
    color_space_field.set_name("color_space_field")

    fc_int_to_color_space_projection = DynamicField.Projection(find_color.get_intention_field().get_dimensionality(), color_space_field_dimensionality, set([0]), [2])
    fc_int_to_color_space_weight = DynamicField.Weight(6.0)
    DynamicField.connect(find_color.get_intention_field(), color_space_field, [fc_int_to_color_space_weight, fc_int_to_color_space_projection])

    color_space_to_fc_cos_projection = DynamicField.Projection(color_space_field_dimensionality, find_color.get_cos_field().get_dimensionality(), set([2]), [0])
    color_space_to_fc_cos_weight = DynamicField.Weight(8.0)
    DynamicField.connect(color_space_field, find_color.get_cos_field(), [color_space_to_fc_cos_projection, color_space_to_fc_cos_weight])

    # create "camera" field
    camera_field = CameraField.NaoCameraField()
    camera_field.set_name("camera_field")
    camera_field_sizes = camera_field.get_output_dimension_sizes()

    camera_to_color_space_weight = DynamicField.Weight(4.0)
    DynamicField.connect(camera_field, color_space_field, [camera_to_color_space_weight])

    # create "spatial target location" field
    spatial_target_field_dimensionality = 2
    spatial_target_kernel = Kernel.GaussKernel(spatial_target_field_dimensionality)
    spatial_target_kernel = Kernel.GaussKernel(spatial_target_field_dimensionality)
    spatial_target_kernel.add_mode(5.0, [1.0] * spatial_target_field_dimensionality, [0.0] * spatial_target_field_dimensionality)
    spatial_target_kernel.calculate()

    spatial_target_field_sizes = move_ee_field_sizes
    spatial_target_field = DynamicField.DynamicField([[spatial_target_field_sizes[0]], [spatial_target_field_sizes[1]]], [], spatial_target_kernel)
    spatial_target_field.set_global_inhibition(400.0)
    spatial_target_field.set_name("spatial_target_field")

    color_space_to_spatial_target_projection = DynamicField.Projection(color_space_field_dimensionality, spatial_target_field_dimensionality, set([0, 1]), [0, 1])
    color_space_to_spatial_target_weight = DynamicField.Weight(10.0)
    DynamicField.connect(color_space_field, spatial_target_field, [color_space_to_spatial_target_projection, color_space_to_spatial_target_weight])

    spatial_target_to_move_ee_int_weight = DynamicField.Weight(5.0)
    DynamicField.connect(spatial_target_field, move_ee.get_intention_field(), [spatial_target_to_move_ee_int_weight])

    # create perception field in end effector space
    perception_ee_field_dimensionality = 2
    perception_ee_kernel = Kernel.GaussKernel(perception_ee_field_dimensionality)
    perception_ee_kernel = Kernel.GaussKernel(perception_ee_field_dimensionality)
    perception_ee_kernel.add_mode(15.0, [1.0] * perception_ee_field_dimensionality, [0.0] * perception_ee_field_dimensionality)
    perception_ee_kernel.calculate()

    perception_ee_field_sizes = move_ee_field_sizes
    perception_ee_field = DynamicField.DynamicField([[perception_ee_field_sizes[0]], [perception_ee_field_sizes[1]]], [], perception_ee_kernel)
    perception_ee_field.set_global_inhibition(400.0)
    perception_ee_field.set_name("perception_ee_field")

    perception_ee_to_move_ee_cos_weight = DynamicField.Weight(3.5)
    DynamicField.connect(perception_ee_field, move_ee.get_cos_field(), [perception_ee_to_move_ee_cos_weight])

    # create end effector control connectable
    end_effector_control = EndEffectorControl.EndEffectorControl(move_ee_field_sizes, head_speed_fraction = 0.3)
    DynamicField.connect(move_ee.get_intention_field(), end_effector_control)



    time_steps = 300

    task_node_activation = [0] * time_steps

    find_color_intention_node_activation = [0] * time_steps
    find_color_cos_node_activation = [0] * time_steps
    find_color_cos_memory_node_activation = [0] * time_steps
    find_color_intention_field_activation = numpy.zeros((time_steps, find_color_field_size))
    find_color_cos_field_activation = numpy.zeros((time_steps, find_color_field_size))

    move_ee_intention_node_activation = [0] * time_steps
    move_ee_cos_node_activation = [0] * time_steps
    move_ee_cos_memory_node_activation = [0] * time_steps
    move_ee_intention_field_activation = numpy.zeros((time_steps, move_ee_field_sizes[0]))
    move_ee_cos_field_activation = numpy.zeros((time_steps, move_ee_field_sizes[0]))

    gripper_open_intention_node_activation = [0] * time_steps
    gripper_open_cos_node_activation = [0] * time_steps
    gripper_open_cos_memory_node_activation = [0] * time_steps

    gripper_close_intention_node_activation = [0] * time_steps
    gripper_close_cos_node_activation = [0] * time_steps
    gripper_close_cos_memory_node_activation = [0] * time_steps

    gripper_intention_field_activation = numpy.zeros((time_steps, gripper_field_size))
    gripper_cos_field_activation = numpy.zeros((time_steps, gripper_field_size))

    gripper_open_precondition_node_activation = [0] * time_steps
    gripper_close_precondition_node_activation = [0] * time_steps

    color_space_field_x_activation = numpy.zeros((time_steps, color_space_field_sizes[0]))
    color_space_field_y_activation = numpy.zeros((time_steps, color_space_field_sizes[1]))
    color_space_field_hue_activation = numpy.zeros((time_steps, color_space_field_sizes[2]))

    camera_field_x_activation = numpy.zeros((time_steps, camera_field_sizes[0]))
    camera_field_y_activation = numpy.zeros((time_steps, camera_field_sizes[1]))
    camera_field_hue_activation = numpy.zeros((time_steps, camera_field_sizes[2]))

    spatial_target_field_x_activation = numpy.zeros((time_steps, spatial_target_field_sizes[0]))
    spatial_target_field_y_activation = numpy.zeros((time_steps, spatial_target_field_sizes[1]))

    perception_ee_field_x_activation = numpy.zeros((time_steps, perception_ee_field_sizes[0]))



    gripper_boost = math_tools.gauss_1d(gripper_field_size, amplitude=10.0, sigma=2.0, shift=gripper_field_size-5) 
    gripper_open.get_cos_field().set_boost(gripper_boost)

#    perception_ee_boost = math_tools.gauss_2d(perception_ee_field_sizes, amplitude=8.0, sigmas=[2.0, 2.0], shifts=[20,5])
#    perception_ee_field.set_boost(perception_ee_boost)

    for i in range(time_steps):
        print "time step: ", str(i)

#        if (i == 150):
#            camera_boost_0 = math_tools.gauss_3d(camera_field_sizes, amplitude=9.5, sigmas=[2.0, 2.0, 2.0], shifts=[10,30,10])
#            camera_boost_1 = math_tools.gauss_3d(camera_field_sizes, amplitude=9.0, sigmas=[2.0, 2.0, 2.0], shifts=[40,25,30])
#            camera_boost_2 = math_tools.gauss_3d(camera_field_sizes, amplitude=9.0, sigmas=[2.0, 2.0, 2.0], shifts=[30,40,45])
#            camera_boost = camera_boost_0 + camera_boost_1 + camera_boost_2
#            camera_field.set_boost(camera_boost)
#        if (i == 5):
#            camera_field.start_activation_log()
#            camera_field.write_activation_log()
#            camera_field.stop_activation_log()

        if (i == 380):
            fields = [find_color.get_intention_field(),
                      find_color.get_cos_field(),
                      move_ee.get_intention_field(),
                      move_ee.get_cos_field(),
                      gripper_intention_field,
                      gripper_cos_field,
                      spatial_target_field,
                      perception_ee_field,
                      color_space_field]
            for field in fields:
                field.start_activation_log()
                field.write_activation_log()
                field.stop_activation_log()

        if (i == 400):
            perception_ee_boost = math_tools.gauss_2d(perception_ee_field_sizes, amplitude=8.0, sigmas=[2.0, 2.0], shifts=[5,25])
            perception_ee_field.set_boost(perception_ee_boost)


        # step all connectables and behaviors
        task_node.step()
        find_color.step()
        camera_field.step()
        perception_ee_field.step()
        color_space_field.step()
        spatial_target_field.step()
        gripper_intention_field.step()
        gripper_cos_field.step()
        gripper_open.step()
        gripper_close.step()
        gripper_open_precondition_node.step()
        move_ee.step()
        end_effector_control.step()
        gripper_close_precondition_node.step()


        # save task node activation
        task_node_activation[i] = task_node.get_activation()[0]

        # save find color activations
        find_color_intention_node_activation[i] = find_color.get_intention_node().get_activation()[0]
        find_color_cos_node_activation[i] = find_color.get_cos_node().get_activation()[0]
        find_color_cos_memory_node_activation[i] = find_color.get_cos_memory_node().get_activation()[0]
        find_color_intention_field_activation[i] = find_color.get_intention_field().get_activation()
        find_color_cos_field_activation[i] = find_color.get_cos_field().get_activation()

        # save move end effector activations
        move_ee_intention_node_activation[i] = move_ee.get_intention_node().get_activation()[0]
        move_ee_cos_node_activation[i] = move_ee.get_cos_node().get_activation()[0]
        move_ee_cos_memory_node_activation[i] = move_ee.get_cos_memory_node().get_activation()[0]
        move_ee_intention_field_activation[i] = move_ee.get_intention_field().get_activation().max(1)
        move_ee_cos_field_activation[i] = move_ee.get_cos_field().get_activation().max(1)

        # save gripper open activations
        gripper_open_intention_node_activation[i] = gripper_open.get_intention_node().get_activation()[0]
        gripper_open_cos_node_activation[i] = gripper_open.get_cos_node().get_activation()[0]
        gripper_open_cos_memory_node_activation[i] = gripper_open.get_cos_memory_node().get_activation()[0]

        # save gripper close activations
        gripper_close_intention_node_activation[i] = gripper_close.get_intention_node().get_activation()[0]
        gripper_close_cos_node_activation[i] = gripper_close.get_cos_node().get_activation()[0]
        gripper_close_cos_memory_node_activation[i] = gripper_close.get_cos_memory_node().get_activation()[0]

        # save gripper field activations
        gripper_intention_field_activation[i] = gripper_intention_field.get_activation()
        gripper_cos_field_activation[i] = gripper_cos_field.get_activation()

        # save precondition activations
        gripper_open_precondition_node_activation[i] = gripper_open_precondition_node.get_activation()[0]
        gripper_close_precondition_node_activation[i] = gripper_close_precondition_node.get_activation()[0]

        # save color space field activations
        color_space_field_hue_x_activation = color_space_field.get_activation().max(1)
        color_space_field_hue_activation[i] = color_space_field_hue_x_activation.max(0)
        color_space_field_x_activation[i] = color_space_field_hue_x_activation.max(1)
        color_space_field_y_activation[i] = color_space_field.get_activation().max(0).max(1)

        # save camera field activations
        camera_field_hue_x_activation = camera_field.get_activation().max(1)
        camera_field_hue_activation[i] = camera_field_hue_x_activation.max(0)
        camera_field_x_activation[i] = camera_field_hue_x_activation.max(1)
        camera_field_y_activation[i] = camera_field.get_activation().max(0).max(1)

        # save spatial target activations
        spatial_target_field_x_activation[i] = spatial_target_field.get_activation().max(1)
        spatial_target_field_y_activation[i] = spatial_target_field.get_activation().max(0)

        # save perception end effector activations
        perception_ee_field_x_activation[i] = perception_ee_field.get_activation().max(1)

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
    grid[0].set_yticks(range(0,find_color_field_size+10,20))
    grid[0].set_ylabel(r'fc int')

    grid[1].imshow(numpy.rollaxis(find_color_cos_field_activation, 1), aspect="auto", vmin=-10, vmax=10)
    grid[1].invert_yaxis()
    grid[1].set_yticks(range(0,find_color_field_size+10,20))
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
    grid[0].set_yticks(range(0,move_ee_field_sizes[0]+10,20))
    grid[0].set_ylabel(r'mee int')

    grid[1].imshow(numpy.rollaxis(move_ee_cos_field_activation, 1), aspect="auto", vmin=-10, vmax=10)
    grid[1].invert_yaxis()
    grid[1].set_yticks(range(0,move_ee_field_sizes[0]+10,20))
    grid[1].set_ylabel(r'mee cos')

    grid[2].imshow(numpy.rollaxis(spatial_target_field_x_activation, 1), aspect="auto", vmin=-10, vmax=10)
    grid[2].invert_yaxis()
    grid[2].set_yticks(range(0,spatial_target_field_sizes[0]+10,20))
    grid[2].set_ylabel(r'st x')

    grid[3].imshow(numpy.rollaxis(spatial_target_field_y_activation, 1), aspect="auto", vmin=-10, vmax=10)
    grid[3].invert_yaxis()
    grid[3].set_yticks(range(0,spatial_target_field_sizes[1]+10,20))
    grid[3].set_ylabel(r'st y')

    grid[4].imshow(numpy.rollaxis(perception_ee_field_x_activation, 1), aspect="auto", vmin=-10, vmax=10)
    grid[4].invert_yaxis()
    grid[4].set_yticks(range(0,perception_ee_field_sizes[0]+10,20))
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
    grid[0].set_yticks(range(0,gripper_field_size+10,20))
    grid[0].set_ylabel(r'go int')

    grid[1].imshow(numpy.rollaxis(gripper_cos_field_activation, 1), aspect="auto", vmin=-10, vmax=10)
    grid[1].invert_yaxis()
    grid[1].set_yticks(range(0,gripper_field_size+10,20))
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
    grid[0].set_yticks(range(0,color_space_field_sizes[0]+10,20))
    grid[0].set_ylabel(r'cs x')

    grid[1].imshow(numpy.rollaxis(color_space_field_y_activation, 1), aspect="auto", vmin=-10, vmax=10)
    grid[1].invert_yaxis()
    grid[1].set_yticks(range(0,color_space_field_sizes[1]+10,20))
    grid[1].set_ylabel(r'cs y')

    grid[2].imshow(numpy.rollaxis(color_space_field_hue_activation, 1), aspect="auto", vmin=-10, vmax=10)
    grid[2].invert_yaxis()
    grid[2].set_yticks(range(0,color_space_field_sizes[2]+10,20))
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
    grid[0].set_yticks(range(0,camera_field_sizes[0]+10,20))
    grid[0].set_ylabel(r'cam x')

    grid[1].imshow(numpy.rollaxis(camera_field_y_activation, 1), aspect="auto", vmin=-10, vmax=10)
    grid[1].invert_yaxis()
    grid[1].set_yticks(range(0,camera_field_sizes[1]+10,20))
    grid[1].set_ylabel(r'cam y')

    grid[2].imshow(numpy.rollaxis(camera_field_hue_activation, 1), aspect="auto", vmin=-10, vmax=10)
    grid[2].invert_yaxis()
    grid[2].set_yticks(range(0,camera_field_sizes[2]+10,20))
    grid[2].set_ylabel(r'cam hue')

    grid[2].set_xlabel(r'time steps')
    grid[2].set_xticks(range(0,time_steps+100,200))

    plt.savefig("fig/camera.pdf", format="pdf")


    plt.show()

#    act = eb0_intention_field_activation[500]
#    x,y = numpy.mgrid[0:act.shape[0]:1, 0:act.shape[1]:1]
#    s = mlab.surf(x, y, act)
#    mlab.show()

if __name__ == "__main__":
    main()
