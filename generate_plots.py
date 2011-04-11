import numpy
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from enthought.mayavi import mlab
from mpl_toolkits.axes_grid import ImageGrid
from mpl_toolkits.axes_grid import make_axes_locatable
import plot_settings

from matplotlib import cm

def main():

    fc_int_file = open("snapshots/find_color_intention_field.txt", 'r')
    fc_int_field = numpy.fromfile(fc_int_file, sep=', ')
    fc_cos_file = open("snapshots/find_color_cos_field.txt", 'r')
    fc_cos_field = numpy.fromfile(fc_cos_file, sep=', ')

    mee_int_file = open("snapshots/move_ee_intention_field.txt", 'r')
    mee_int_field = numpy.fromfile(mee_int_file, sep=', ')
    mee_int_field = mee_int_field.reshape(50,50)
    mee_cos_file = open("snapshots/move_ee_cos_field.txt", 'r')
    mee_cos_field = numpy.fromfile(mee_cos_file, sep=', ')
    mee_cos_field = mee_cos_field.reshape(50,50)

    gr_int_file = open("snapshots/gripper_intention_field.txt", 'r')
    gr_int_field = numpy.fromfile(gr_int_file, sep=', ')
    gr_cos_file = open("snapshots/gripper_cos_field.txt", 'r')
    gr_cos_field = numpy.fromfile(gr_cos_file, sep=', ')

    st_file = open("snapshots/spatial_target_field.txt", 'r')
    st_field = numpy.fromfile(st_file, sep=', ')
    st_field = st_field.reshape(50,50)

    pe_file = open("snapshots/perception_ee_field.txt", 'r')
    pe_field = numpy.fromfile(pe_file, sep=', ')
    pe_field = pe_field.reshape(50,50)

    plot_settings.set_mode("mini")

    ##########################################################################
    # create a figure for the find color int plot
    fig = plt.figure(1)
#    fig.subplots_adjust(bottom=0.07, left=0.07, right=0.97, top=0.93)

    plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])
    plt.grid(color='grey', linestyle='dotted')

    plt.plot(fc_int_field, 'r-', label=r'fc int', antialiased=True)
    plt.xlabel(r'hue')
    plt.ylabel(r'act')

    plt.savefig("fig/find_color_int_field.pdf", format="pdf")

    ##########################################################################
    # create a figure for the find color cos plot
    fig = plt.figure(2)
#    fig.subplots_adjust(bottom=0.07, left=0.07, right=0.97, top=0.93)

    plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])
    plt.grid(color='grey', linestyle='dotted')

    plt.plot(fc_cos_field, 'r-', label=r'fc cos', antialiased=True)
    plt.xlabel(r'hue')
    plt.ylabel(r'act')

    plt.savefig("fig/find_color_cos_field.pdf", format="pdf")

    ##########################################################################
    # create a figure for the gripper int plot
    fig = plt.figure(3)
#    fig.subplots_adjust(bottom=0.07, left=0.07, right=0.97, top=0.93)

    plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])
    plt.grid(color='grey', linestyle='dotted')

    plt.plot(gr_int_field, 'r-', label=r'gr int', antialiased=True)
    plt.xlabel(r'$\Delta g$')
    plt.ylabel(r'act')

    plt.savefig("fig/gripper_int_field.pdf", format="pdf")

    ##########################################################################
    # create a figure for the gripper cos plot
    fig = plt.figure(4)
#    fig.subplots_adjust(bottom=0.07, left=0.07, right=0.97, top=0.93)

    plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])
    plt.grid(color='grey', linestyle='dotted')

    plt.plot(gr_cos_field, 'r-', label=r'gr cos', antialiased=True)
    plt.xlabel(r'$\Delta g$')
    plt.ylabel(r'act')

    plt.savefig("fig/gripper_cos_field.pdf", format="pdf")

    ##########################################################################
    # create a figure for the move ee int plot
    x,y = numpy.mgrid[0:mee_int_field.shape[0]:1, 0:mee_int_field.shape[1]:1]

    fig = plt.figure(5)
#    fig.subplots_adjust(bottom=0.07, left=0.07, right=0.97, top=0.93)

    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, mee_int_field, rstride=1, cstride=1, vmin=-5, vmax=5, cmap=cm.jet, linewidth=0, antialiased=True)
    ax.set_zlim3d(-5.01, 5.01)

#    ax.set_xticks([0,20,40])
#    ax.set_yticks([0,20,40])
#    ax.w_zaxis.set_ticks([-4,0,4])

#    ax.set_xlabel(r'x')
#    ax.set_ylabel(r'y')
#    ax.set_zlabel(r'act')

    plt.savefig("fig/move_ee_int_field.pdf", format="pdf")

    ##########################################################################
    # create a figure for the move ee cos plot
    x,y = numpy.mgrid[0:mee_cos_field.shape[0]:1, 0:mee_cos_field.shape[1]:1]

    fig = plt.figure(6)
#    fig.subplots_adjust(bottom=0.07, left=0.07, right=0.97, top=0.93)

    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, mee_cos_field, rstride=1, cstride=1, vmin=-5, vmax=5, cmap=cm.jet, linewidth=0, antialiased=True)
    ax.set_zlim3d(-5.01, 5.01)

#    ax.set_xticks([0,20,40])
#    ax.set_yticks([0,20,40])
#    ax.w_zaxis.set_ticks([-4,0,4])

#    ax.set_xlabel(r'x')
#    ax.set_ylabel(r'y')
#    ax.set_zlabel(r'act')

    plt.savefig("fig/move_ee_cos_field.pdf", format="pdf")

    ##########################################################################
    # create a figure for the spatial target plot
    x,y = numpy.mgrid[0:st_field.shape[0]:1, 0:st_field.shape[1]:1]

    fig = plt.figure(7)
#    fig.subplots_adjust(bottom=0.07, left=0.07, right=0.97, top=0.93)

    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, st_field, rstride=1, cstride=1, vmin=-5, vmax=5, cmap=cm.jet, linewidth=0, antialiased=True)
    ax.set_zlim3d(-5.01, 5.01)

#    ax.set_xticks([0,20,40])
#    ax.set_yticks([0,20,40])
#    ax.w_zaxis.set_ticks([-4,0,4])

#    ax.set_xlabel(r'x')
#    ax.set_ylabel(r'y')
#    ax.set_zlabel(r'act')

    plt.savefig("fig/spatial_target_field.pdf", format="pdf")

    ##########################################################################
    # create a figure for the perception ee plot
    x,y = numpy.mgrid[0:pe_field.shape[0]:1, 0:pe_field.shape[1]:1]

    fig = plt.figure(8)
#    fig.subplots_adjust(bottom=0.07, left=0.07, right=0.97, top=0.93)

    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, pe_field, rstride=1, cstride=1, vmin=-5, vmax=5, cmap=cm.jet, linewidth=0, antialiased=True)
    ax.set_zlim3d(-5.01, 5.01)

#    ax.set_xticks([0,20,40])
#    ax.set_yticks([0,20,40])
#    ax.w_zaxis.set_ticks([-4,0,4])

#    ax.set_xlabel(r'x')
#    ax.set_ylabel(r'y')
#    ax.set_zlabel(r'act')

    plt.savefig("fig/perception_ee_field.pdf", format="pdf")


    plt.show()


    fc_int_file.close()
    fc_cos_file.close()
    mee_int_file.close()
    mee_cos_file.close()
    gr_int_file.close()
    gr_cos_file.close()
    st_file.close()
    pe_file.close()





if __name__ == "__main__":
    main()
