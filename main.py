import BehavioralOrganization as BehOrg
import DynamicField
import numpy
import math
import matplotlib.pyplot as plt
from enthought.mayavi import mlab
from mpl_toolkits.axes_grid import ImageGrid
from mpl_toolkits.axes_grid import make_axes_locatable
import plot_settings

def gauss_2d(sizes, amplitude, sigmas, shifts):
    assert len(sizes) == 2, "The sizes variable has to contain exactly two values."
    assert len(sigmas) == 2, "The sigmas variable has to contain exactly two values."
    assert len(shifts) == 2, "The shifts variable has to contain exactly two values."
    
    gauss = numpy.zeros(sizes)
    for i in range(sizes[0]):
        for j in range(sizes[1]):
            gauss[i][j] = amplitude * math.exp(- (math.pow(i - shifts[0], 2.0)/(2*math.pow(sigmas[0], 2.0)) + math.pow(j - shifts[1], 2.0)/(2*math.pow(sigmas[1], 2.0))))

    return gauss

    

def main():
    task_node = DynamicField.DynamicField([], [], None)

    field_sizes = [80, 80]

    int_weight_0 = gauss_2d(field_sizes, amplitude=10, sigmas=[5.0, 5.0], shifts=[20, 20])
    int_weight_1 = gauss_2d(field_sizes, amplitude=10, sigmas=[5.0, 5.0], shifts=[40, 50])

    elem_behavior_0 = BehOrg.ElementaryBehavior(field_dimensionality=2,
                                                field_sizes=[[field_sizes[0]],[field_sizes[1]]],
                                                field_resolutions=[],
                                                int_node_to_int_field_weight=int_weight_0,
                                                int_field_to_cos_field_weight=3.5,
                                                cos_field_to_cos_node_weight=5.5,
                                                cos_node_to_cos_memory_node_weight=2.5,
                                                int_inhibition_weight=-6.0,
                                                reactivating=False)

    elem_behavior_1 = BehOrg.ElementaryBehavior(field_dimensionality=2,
                                                field_sizes=[[field_sizes[0]],[field_sizes[1]]],
                                                field_resolutions=[],
                                                int_node_to_int_field_weight=int_weight_1,
                                                int_field_to_cos_field_weight=3.5,
                                                cos_field_to_cos_node_weight=5.5,
                                                cos_node_to_cos_memory_node_weight=2.5,
                                                int_inhibition_weight=-6.0,
                                                reactivating=False)

    BehOrg.connect_to_task(task_node, elem_behavior_0)
    BehOrg.connect_to_task(task_node, elem_behavior_1)

    competition_nodes = BehOrg.competition(elem_behavior_0, elem_behavior_1, task_node, bidirectional=True)


    time_steps = 1200

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

        if (i > 50):
            task_node.set_boost(11)
        if (i > 300):
            elem_behavior_0.get_cos_field().set_boost(1.0)
        if (i > 500):
            elem_behavior_0.get_cos_field().set_boost(0.0)
        if (i > 700):
            elem_behavior_1.get_cos_field().set_boost(1.0)
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
#    fig.subplots_adjust(bottom=0.1, left=0.1, right=0.95, top=0.95)
    fig.subplots_adjust(bottom=0.1)

#    plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])

    time_course_subplot = plt.subplot(2,1,1)
    time_course_subplot.axes.grid(color='grey', linestyle='dashed')

    plt.xlabel("time steps")
    plt.ylabel("activation")
    
    plt.plot(task_node_activation, 'k-', label='task')

    plt.plot(eb0_intention_node_activation, 'r-', linewidth=1.0, label='EB0 intention', antialiased=True)
    plt.plot(eb0_cos_node_activation, 'b-', linewidth=1.0, label='EB0 cos', antialiased=True)
    plt.plot(eb0_cos_memory_node_activation, 'c-', label='EB0 cos mem', antialiased=True)

    plt.plot(eb1_intention_node_activation, 'r--', linewidth=1.0, label='EB1 intention', antialiased=True)
    plt.plot(eb1_cos_node_activation, 'b--', linewidth=1.0, label='EB1 cos', antialiased=True)
    plt.plot(eb1_cos_memory_node_activation, 'c--', label='EB1 cos mem', antialiased=True)

    plt.plot(competition_node_01_activation, 'g-.', label='competition 01', antialiased=True)
    plt.plot(competition_node_10_activation, 'm-.', label='competition 10', antialiased=True)
    plt.legend(loc='upper right')

    grid = ImageGrid(fig, 212, nrows_ncols = (4,1), axes_pad=0.1)

    grid[0].imshow(numpy.rollaxis(eb0_intention_field_activation_1d, 1), label='eb0 int field', aspect="auto")
    grid[0].invert_yaxis()
    grid[0].set_yticks(range(0,90,20))
    grid[0].set_ylabel("EB0 int")

    grid[1].imshow(numpy.rollaxis(eb0_cos_field_activation_1d, 1), label='eb0 cos field', aspect="auto")
    grid[1].invert_yaxis()
    grid[1].set_yticks(range(0,90,20))
    grid[1].set_ylabel("EB0 cos")

    grid[2].imshow(numpy.rollaxis(eb1_intention_field_activation_1d, 1), label='eb1 int field', aspect="auto")
    grid[2].invert_yaxis()
    grid[2].set_yticks(range(0,90,20))
    grid[2].set_ylabel("EB1 int")

    grid[3].imshow(numpy.rollaxis(eb1_cos_field_activation_1d, 1), label='eb1 cos field', aspect="auto")
    grid[3].invert_yaxis()
    grid[3].set_yticks(range(0,90,20))
    grid[3].set_ylabel("EB1 cos")
    grid[3].set_xlabel("time steps")
    grid[3].set_xticks(range(0,1300,200))

    plt.savefig("competition_plot.eps")
    plt.show()

#    act = eb0_intention_field_activation[500]
#    x,y = numpy.mgrid[0:act.shape[0]:1, 0:act.shape[1]:1]
#    s = mlab.surf(x, y, act)
#    mlab.show()

if __name__ == "__main__":
    main()
