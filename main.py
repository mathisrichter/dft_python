import BehavioralOrganization as BehOrg
import DynamicField
import numpy
import math
import matplotlib.pyplot as plt
from enthought.mayavi import mlab
from mpl_toolkits.axes_grid import ImageGrid
from mpl_toolkits.axes_grid import make_axes_locatable
import plot_settings

def main():
    task_node = DynamicField.DynamicField([], [], None)

    field_sizes = [50, 50]

    int_weight = numpy.zeros(field_sizes)
    for i in range(field_sizes[0]):
        for j in range(field_sizes[1]):
            int_weight[i][j] = 10 * math.exp(- (math.pow(i - 17, 2.0)/(2*math.pow(5.0, 2.0)) + math.pow(j - 17, 2.0)/(2*math.pow(5.0, 2.0))))

    elem_behavior_0 = BehOrg.ElementaryBehavior(field_dimensionality=2,
                                                field_sizes=[[field_sizes[0]],[field_sizes[1]]],
                                                field_resolutions=[],
                                                int_node_to_int_field_weight=int_weight,
                                                int_field_to_cos_field_weight=3.5,
                                                cos_field_to_cos_node_weight=5.5,
                                                cos_node_to_cos_memory_node_weight=2.5,
                                                int_inhibition_weight=-6.0,
                                                reactivating=False)

    elem_behavior_1 = BehOrg.ElementaryBehavior(field_dimensionality=2,
                                                field_sizes=[[field_sizes[0]],[field_sizes[1]]],
                                                field_resolutions=[],
                                                int_node_to_int_field_weight=numpy.transpose(int_weight),
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

        if (i > 20):
            task_node.set_boost(11)
        if (i > 200):
            elem_behavior_0.get_cos_field().set_boost(1.5)
        if (i > 400):
            elem_behavior_0.get_cos_field().set_boost(0.0)
        if (i > 600):
            elem_behavior_1.get_cos_field().set_boost(1.5)
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

#    plot_settings.set_mode("default")
    fig = plt.figure(1, (5,1))
    grid = ImageGrid(fig, 111, nrows_ncols = (5,1), axes_pad=0.1, cbar_mode="single", cbar_size="0.4%", aspect=False)
    
    plt.axes([0.125,0.2,0.95-0.125,0.95-0.3])
    plt.xlabel(r'time steps')
    plt.ylabel(r'activation $u$')
    plt.title(r'Competing elementary behaviors')

    time_course_subplot = plt.subplot(5,1,1)
    
    plt.plot(task_node_activation, label='task')
    plt.plot(eb0_intention_node_activation, label='eb0 int')
    plt.plot(eb0_cos_node_activation, label='eb0 cos')
    plt.plot(eb0_cos_memory_node_activation, label='eb0 cos mem')
    plt.plot(eb1_intention_node_activation, label='eb1 int')
    plt.plot(eb1_cos_node_activation, label='eb1 cos')
    plt.plot(eb1_cos_memory_node_activation, label='eb1 cos mem')
    plt.plot(competition_node_01_activation, label='comp 01')
    plt.plot(competition_node_10_activation, label='comp 10')
    plt.legend(loc='upper right')

    int_field_subplot = plt.subplot(5,1,2)
    im = plt.imshow(numpy.transpose(eb0_intention_field_activation_1d), label='eb0 int field')
#    colorbar = grid.cbar_axes[0].colorbar(im, ticks=[-10, -1, 0, 1, 10])
#    colorbar.ax.set_yticklabels(['sub', 'low', 'zero', 'med', 'high'])
    cos_field_subplot = plt.subplot(5,1,3)
    plt.imshow(numpy.transpose(eb0_cos_field_activation_1d), label='eb0 cos field')

    int_field_subplot = plt.subplot(5,1,4)
    plt.imshow(numpy.transpose(eb1_intention_field_activation_1d), label='eb1 int field')
    cos_field_subplot = plt.subplot(5,1,5)
    plt.imshow(numpy.transpose(eb1_cos_field_activation_1d), label='eb1 cos field')

    plt.savefig("competition_plot.eps")
    plt.show()

#    act = eb0_intention_field_activation[500]
#    x,y = numpy.mgrid[0:act.shape[0]:1, 0:act.shape[1]:1]
#    s = mlab.surf(x, y, act)
#    mlab.show()

if __name__ == "__main__":
    main()
