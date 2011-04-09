import BehavioralOrganization as BehOrg
import DynamicField
import numpy
import math
import matplotlib.pyplot as plt
from matplotlib import rc
from enthought.mayavi import mlab
from mpl_toolkits.axes_grid import ImageGrid
from mpl_toolkits.axes_grid import make_axes_locatable
import plot_settings
import math_tools


def main():
    # create a task node
    task_node = DynamicField.DynamicField([], [], None)

    # create elementary behavior: find color
    find_color_field_size = 50
    find_color_int_weight = gauss_1d(find_color_field_size, amplitude=10, sigma=5.0, shift=10)

    find_color = BehOrg.ElementaryBehavior.with_internal_fields(field_dimensionality=1,
                                                field_sizes=[[field_color_field_size]],
                                                field_resolutions=[],
                                                int_node_to_int_field_weight=find_color_int_weight,
                                                int_field_to_cos_field_weight=3.5,
                                                cos_field_to_cos_node_weight=5.5,
                                                cos_node_to_cos_memory_node_weight=2.5,
                                                int_inhibition_weight=-6.0,
                                                reactivating=False)

    # create elementary behavior: move end effector
    move_ee_field_sizes = [50, 50]
    move_ee_int_weight = numpy.ones((move_ee_field_sizes)) * 2.0
    move_ee = BehOrg.ElementaryBehavior(field_dimensionality=2,
                                                field_sizes=[[move_ee_field_sizes[0]],[move_ee_field_sizes[1]]],
                                                field_resolutions=[],
                                                int_node_to_int_field_weight=move_ee_int_weight,
                                                int_field_to_cos_field_weight=3.5,
                                                cos_field_to_cos_node_weight=5.5,
                                                cos_node_to_cos_memory_node_weight=2.5,
                                                int_inhibition_weight=-6.0,
                                                reactivating=False)

    # create gripper intention and cos fields
    gripper_field_dimensionality = 1
    gripper_field_size = 50

    # gripper intention field and its kernel
    intention_field_kernel = Kernel.GaussKernel(gripper_field_dimensionality)
    intention_field_kernel.add_mode(1.0, [0.5] * gripper_field_dimensionality, [0.0] * gripper_field_dimensionality)
    intention_field_kernel.add_mode(-5.5, [5.5] * gripper_field_dimensionality, [0.0] * gripper_field_dimensionality)
    intention_field_kernel.calculate()
    gripper_intention_field = DynamicField.DynamicField([[gripper_field_size]], [], intention_field_kernel)
    gripper_intention_field.set_global_inhibition(0.5)

    # gripper CoS field and its kernel
    cos_field_kernel = Kernel.GaussKernel(gripper_field_dimensionality)
    cos_field_kernel.add_mode(1.0, [0.5] * gripper_field_dimensionality, [0.0] * gripper_field_dimensionality)
    cos_field_kernel.add_mode(-5.5, [5.5] * gripper_field_dimensionality, [0.0] * gripper_field_dimensionality)
    cos_field_kernel.calculate()
    gripper_cos_field = DynamicField.DynamicField([[gripper_field_size]], [], cos_field_kernel)
    gripper_cos_field.set_global_inhibition(0.5)

    gripper_close_int_weight = gauss_1d(gripper_field_size, amplitude=10, sigma=5.0, shift=10)
    gripper_close = BehOrg.ElementaryBehavior(intention_field=gripper_intention_field,
                                              cos_field=gripper_cos_field,
                                              int_node_to_int_field_weight=gripper_close_int_weight,
                                              int_field_to_cos_field_weight=3.5,
                                              cos_field_to_cos_node_weight=5.5,
                                              cos_node_to_cos_memory_node_weight=2.5,
                                              int_inhibition_weight=-6.0,
                                              reactivating=False)

    # create elementary behavior: gripper close
    gripper_close_int_weight = gauss_1d(gripper_field_size, amplitude=10, sigma=5.0, shift=5)
    gripper_close = BehOrg.ElementaryBehavior(intention_field=gripper_intention_field,
                                              cos_field=gripper_cos_field,
                                              int_node_to_int_field_weight=gripper_close_int_weight,
                                              int_field_to_cos_field_weight=3.5,
                                              cos_field_to_cos_node_weight=5.5,
                                              cos_node_to_cos_memory_node_weight=2.5,
                                              int_inhibition_weight=-6.0,
                                              reactivating=False)

    # create elementary behavior: gripper open
    gripper_open_int_weight = gauss_1d(gripper_field_size, amplitude=10, sigma=5.0, shift=gripper_field_size-5)
    gripper_open = BehOrg.ElementaryBehavior(intention_field=gripper_intention_field,
                                              cos_field=gripper_cos_field,
                                              int_node_to_int_field_weight=gripper_open_int_weight,
                                              int_field_to_cos_field_weight=3.5,
                                              cos_field_to_cos_node_weight=5.5,
                                              cos_node_to_cos_memory_node_weight=2.5,
                                              int_inhibition_weight=-6.0,
                                              reactivating=False)




    BehOrg.connect_to_task(task_node, elem_behavior_0)
    BehOrg.connect_to_task(task_node, elem_behavior_1)

    competition_nodes = BehOrg.competition(elem_behavior_0, elem_behavior_1, task_node, bidirectional=True)


    time_steps = 1000

    task_node_activation = [0] * time_steps
    eb0_intention_node_activation = [0] * time_steps
    eb0_intention_field_activation = [0] * time_steps
    eb0_cos_node_activation = [0] * time_steps
    eb0_cos_field_activation = [0] * time_steps
    eb0_cos_memory_node_activation = [0] * time_steps
    eb0_intention_field_activation_1d = numpy.zeros((time_steps, field_sizes[1]))
    eb0_cos_field_activation = [0] * time_steps
    eb0_cos_field_activation_1d = numpy.zeros((time_steps, field_sizes[1]))
    competition_node_01_activation = [0] * time_steps

    eb1_intention_node_activation = [0] * time_steps
    eb1_intention_field_activation = [0] * time_steps
    eb1_cos_node_activation = [0] * time_steps
    eb1_cos_field_activation = [0] * time_steps
    eb1_cos_memory_node_activation = [0] * time_steps
    eb1_intention_field_activation_1d = numpy.zeros((time_steps, field_sizes[1]))
    eb1_cos_field_activation = [0] * time_steps
    eb1_cos_field_activation_1d = numpy.zeros((time_steps, field_sizes[1]))
    competition_node_10_activation = [0] * time_steps


    print_output = False

    for i in range(time_steps):

        if (i > 1):
            task_node.set_boost(10)
        if (i > 200):
            elem_behavior_0.get_cos_field().set_boost(1.0)
        if (i > 350):
            elem_behavior_0.get_cos_field().set_boost(0.0)
        if (i > 500):
            elem_behavior_1.get_cos_field().set_boost(1.0)
        if (i > 600):
            elem_behavior_1.get_cos_field().set_boost(0.0)
        task_node.step()
        elem_behavior_0.step()
        competition_nodes[0].step()
        competition_nodes[1].step()
        elem_behavior_1.step()

        if (print_output is True):
            print("task node activation (boost: " + str(task_node.get_boost()) + ")")
        task_node_activation[i] = task_node.get_activation()[0]
        if (print_output is True):
            print(task_node_activation[i])

        if (print_output is True):
            print("int node activation")
        eb0_intention_node_activation[i] = elem_behavior_0.get_intention_node().get_activation()[0]
        if (print_output is True):
            print(eb0_intention_node_activation[i])
        if (print_output is True):
            print("int field activation")
        eb0_intention_field_activation[i] = elem_behavior_0.get_intention_field().get_activation()
        if (print_output is True):
            print(eb0_intention_field_activation[i])
        if (print_output is True):
            print("int field activation 1d")
        eb0_intention_field_activation_1d[i] = eb0_intention_field_activation[i].sum(0)
        if (print_output is True):
            print(eb0_intention_field_activation_1d)
        if (print_output is True):
            print("cos field activation (boost: " + str(elem_behavior_0.get_cos_field().get_boost()) + ")")
        eb0_cos_field_activation[i] = elem_behavior_0.get_cos_field().get_activation()
        if (print_output is True):
            print(eb0_cos_field_activation[i])
        if (print_output is True):
            print("cos field activation 1d: ")
        eb0_cos_field_activation_1d[i] = eb0_cos_field_activation[i].sum(0)
        if (print_output is True):
            print(eb0_cos_field_activation_1d)
        if (print_output is True):
            print("cos node activation")
        eb0_cos_node_activation[i] = elem_behavior_0.get_cos_node().get_activation()[0]
        if (print_output is True):
            print(eb0_cos_node_activation[i])
            print("cos mem node activation")
        eb0_cos_memory_node_activation[i] = elem_behavior_0.get_cos_memory_node().get_activation()[0]
        if (print_output is True):
            print(eb0_cos_memory_node_activation[i])

            print("")
            print("competition node 01 activation")
        competition_node_01_activation[i] = competition_nodes[0].get_activation()[0]
        if (print_output is True):
            print(competition_node_01_activation[i])
            print("competition node 10 activation")
        competition_node_10_activation[i] = competition_nodes[1].get_activation()[0]
        if (print_output is True):
            print(competition_nodes[1].get_activation())
            print("")

            print("int node activation 1")
        eb1_intention_node_activation[i] = elem_behavior_1.get_intention_node().get_activation()[0]
        if (print_output is True):
            print(elem_behavior_1.get_intention_node().get_activation())
            print("int field activation 1")
        eb1_intention_field_activation[i] = elem_behavior_1.get_intention_field().get_activation()
        if (print_output is True):
            print(elem_behavior_1.get_intention_field().get_activation())
            print("int field activation 1 1d")
        eb1_intention_field_activation_1d[i] = eb1_intention_field_activation[i].sum(0)
        if (print_output is True):
            print(eb0_intention_field_activation_1d)
        if (print_output is True):
            print("cos field activation 1 (boost: " + str(elem_behavior_1.get_cos_field().get_boost()) + ")")
        eb1_cos_field_activation[i] = elem_behavior_1.get_cos_field().get_activation()
        if (print_output is True):
            print(elem_behavior_1.get_cos_field().get_activation())
        eb1_cos_field_activation_1d[i] = eb1_cos_field_activation[i].sum(0)
        if (print_output is True):
            print(eb1_cos_field_activation_1d)
        if (print_output is True):
            print("cos node activation 1")
        eb1_cos_node_activation[i] = elem_behavior_1.get_cos_node().get_activation()[0]
        if (print_output is True):
            print(elem_behavior_1.get_cos_node().get_activation())
            print("cos mem node activation 1")
        eb1_cos_memory_node_activation[i] = elem_behavior_1.get_cos_memory_node().get_activation()[0]
        if (print_output is True):
            print(elem_behavior_1.get_cos_memory_node().get_activation())
            print("\n----------------------------------------------------\n")

    plot_settings.set_mode("icdl")

    fig = plt.figure(1)
    fig.subplots_adjust(bottom=0.07, left=0.07, right=0.97, top=0.93)

    plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])

    time_course_subplot = plt.subplot(2,1,1)
    time_course_subplot.axes.grid(color='grey', linestyle='dotted')

    plt.xlabel(r'time steps')
    plt.ylabel(r'activation')
    
    plt.plot(task_node_activation, 'y-', label=r'task')

    plt.plot(eb0_intention_node_activation, 'r-', label=r'EB0 intention', antialiased=True)
    plt.plot(eb0_cos_node_activation, 'b-', label=r'EB0 cos', antialiased=True)
    plt.plot(eb0_cos_memory_node_activation, 'c-', label=r'EB0 cos mem', antialiased=True)

    plt.plot(eb1_intention_node_activation, 'r--', label=r'EB1 intention', antialiased=True)
    plt.plot(eb1_cos_node_activation, 'b--', label=r'EB1 cos', antialiased=True)
    plt.plot(eb1_cos_memory_node_activation, 'c--', label=r'EB1 cos mem', antialiased=True)

    plt.plot(competition_node_01_activation, 'g-.', label=r'competition 01', antialiased=True)
    plt.plot(competition_node_10_activation, 'm-.', label=r'competition 10', antialiased=True)
    plt.legend(loc='upper right')

    plt.annotate('boost', xy=(200,-2), xytext=(50,-10), arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90,rad=10"))

    grid = ImageGrid(fig, 212, nrows_ncols = (4,1), axes_pad=0.1, aspect=False)

    grid[0].imshow(numpy.rollaxis(eb0_intention_field_activation_1d, 1), label='eb0 int field', aspect="auto")
    grid[0].invert_yaxis()
    grid[0].set_yticks(range(0,field_sizes[0]+10,20))
    grid[0].set_ylabel(r'EB0 int')

    grid[1].imshow(numpy.rollaxis(eb0_cos_field_activation_1d, 1), label='eb0 cos field', aspect="auto")
    grid[1].invert_yaxis()
    grid[1].set_yticks(range(0,field_sizes[0]+10,20))
    grid[1].set_ylabel(r'EB0 cos')

    grid[2].imshow(numpy.rollaxis(eb1_intention_field_activation_1d, 1), label='eb1 int field', aspect="auto")
    grid[2].invert_yaxis()
    grid[2].set_yticks(range(0,field_sizes[0]+10,20))
    grid[2].set_ylabel(r'EB1 int')

    grid[3].imshow(numpy.rollaxis(eb1_cos_field_activation_1d, 1), label='eb1 cos field', aspect="auto")
    grid[3].invert_yaxis()
    grid[3].set_yticks(range(0,field_sizes[0]+10,20))
    grid[3].set_ylabel(r'EB1 cos')
    grid[3].set_xlabel(r'time steps')
    grid[3].set_xticks(range(0,time_steps+100,200))

    plt.savefig("competition_plot.pdf", format="pdf")
    plt.show()

#    act = eb0_intention_field_activation[500]
#    x,y = numpy.mgrid[0:act.shape[0]:1, 0:act.shape[1]:1]
#    s = mlab.surf(x, y, act)
#    mlab.show()

if __name__ == "__main__":
    main()
