#!/usr/bin/env python
"""
This plot displays the audio spectrum from the microphone.

Based on updating_plot.py
"""
# Major library imports
import pyaudio
from numpy import zeros, linspace, short, fromstring, hstack, transpose
from scipy import fft
import numpy

import os
os.environ['ETS_TOOLKIT']='qt4'

# Enthought library imports
from enthought.chaco.default_colormaps import jet
from enthought.enable.api import Window, Component, ComponentEditor
from enthought.traits.api import HasTraits, Instance
from enthought.traits.ui.api import Item, Group, View, Handler
from enthought.enable.example_support import DemoFrame, demo_main
from enthought.pyface.timer.api import Timer

# Chaco imports
from enthought.chaco.api import Plot, ArrayPlotData, HPlotContainer


import PyQt4.QtCore as QtCore
import PyQt4.QtGui as QtGui
import PyQt4.Qt as Qt
import sys
import BehavioralOrganization as BehOrg
import DynamicField
import Kernel
import numpy
import math
import math_tools
import CameraField
import EndEffectorControl




#============================================================================
# Create the Chaco plot.
#============================================================================

def _create_plot_component(obj):
    # Spectrogram plot
    spectrogram_data = numpy.zeros((60,45))
    obj.spectrogram_plotdata = ArrayPlotData()
    obj.spectrogram_plotdata.set_data('imagedata', spectrogram_data)
    spectrogram_plot = Plot(obj.spectrogram_plotdata)
    spectrogram_plot.img_plot('imagedata',
                              name='Spectrogram',
                              xbounds=(0, 59),
                              ybounds=(0, 44),
                              colormap=jet,
                              )
    range_obj = spectrogram_plot.plots['Spectrogram'][0].value_mapper.range
    range_obj.high = 10
    range_obj.low = -10
    spectrogram_plot.title = 'Spectrogram'
    obj.spectrogram_plot = spectrogram_plot

    container = HPlotContainer()
    container.add(spectrogram_plot)

    return container


# HasTraits class that supplies the callable for the timer event.
class TimerController(HasTraits):

    def __init__(self):

        #############################################################
        # create a task node
        self._task_node = DynamicField.DynamicField([], [], None)
        self._task_node.set_boost(10)

        # create elementary behavior: find color
        find_color_field_size = 15
        find_color_int_weight = math_tools.gauss_1d(find_color_field_size, amplitude=15.0, sigma=2.0, shift=0)

        self._find_color = BehOrg.ElementaryBehavior.with_internal_fields(field_dimensionality=1,
                                                    field_sizes=[[find_color_field_size]],
                                                    field_resolutions=[],
                                                    int_node_to_int_field_weight=find_color_int_weight,
                                                    name="find color")

        # create elementary behavior: move end effector
        move_ee_field_sizes = [40, 30]
        move_ee_int_weight = numpy.ones((move_ee_field_sizes)) * 4.0
        self._move_ee = BehOrg.ElementaryBehavior.with_internal_fields(field_dimensionality=2,
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
        self._gripper_intention_field = DynamicField.DynamicField([[gripper_field_size]], [], intention_field_kernel)
        self._gripper_intention_field.set_global_inhibition(200.0)

        # gripper CoS field and its kernel
        cos_field_kernel = Kernel.GaussKernel(gripper_field_dimensionality)
        cos_field_kernel.add_mode(15.0, [1.0] * gripper_field_dimensionality, [0.0] * gripper_field_dimensionality)
        cos_field_kernel.calculate()
        self._gripper_cos_field = DynamicField.DynamicField([[gripper_field_size]], [], cos_field_kernel)
        self._gripper_cos_field.set_global_inhibition(200.0)

        # connect the gripper intention and CoS field
        gripper_int_field_to_cos_field_weight = DynamicField.Weight(2.5)
        DynamicField.connect(self._gripper_intention_field, self._gripper_cos_field, [gripper_int_field_to_cos_field_weight])

        # create elementary behavior: gripper close
        gripper_close_int_weight = math_tools.gauss_1d(gripper_field_size, amplitude=15, sigma=2.0, shift=5)
        self._gripper_close = BehOrg.ElementaryBehavior(intention_field=self._gripper_intention_field,
                                                  cos_field=self._gripper_cos_field,
                                                  int_node_to_int_field_weight=gripper_close_int_weight,
                                                  name="gripper close")

        # create elementary behavior: gripper open
        gripper_open_int_weight = math_tools.gauss_1d(gripper_field_size, amplitude=15, sigma=2.0, shift=gripper_field_size-5)
        self._gripper_open = BehOrg.ElementaryBehavior(intention_field=self._gripper_intention_field,
                                                  cos_field=self._gripper_cos_field,
                                                  int_node_to_int_field_weight=gripper_open_int_weight,
                                                  name="gripper open")
        self._gripper_intention_field.set_name("gripper_intention_field")
        self._gripper_cos_field.set_name("gripper_cos_field")


        # connect all elementary behaviors to the task node
        BehOrg.connect_to_task(self._task_node, self._find_color)
        BehOrg.connect_to_task(self._task_node, self._move_ee)
        BehOrg.connect_to_task(self._task_node, self._gripper_open)
        BehOrg.connect_to_task(self._task_node, self._gripper_close)

        # create precondition nodes
        self._gripper_open_precondition_node = BehOrg.precondition(self._gripper_open, self._move_ee, self._task_node)
        self._gripper_close_precondition_node = BehOrg.precondition(self._move_ee, self._gripper_close, self._task_node)

        # create perception color-space field
        color_space_field_dimensionality = 3
        color_space_kernel = Kernel.GaussKernel(color_space_field_dimensionality)
        color_space_kernel.add_mode(5.0, [1.0] * color_space_field_dimensionality, [0.0] * color_space_field_dimensionality)
        color_space_kernel.calculate()

        color_space_field_sizes = [move_ee_field_sizes[0], move_ee_field_sizes[1], find_color_field_size]
        self._color_space_field = DynamicField.DynamicField([[color_space_field_sizes[0]],[color_space_field_sizes[1]],[color_space_field_sizes[2]]], [], color_space_kernel)
        self._color_space_field.set_global_inhibition(400.0)
        self._color_space_field.set_relaxation_time(2.0)
        self._color_space_field.set_name("color_space_field")

        fc_int_to_color_space_projection = DynamicField.Projection(self._find_color.get_intention_field().get_dimensionality(), color_space_field_dimensionality, set([0]), [2])
        fc_int_to_color_space_weight = DynamicField.Weight(6.0)
        DynamicField.connect(self._find_color.get_intention_field(), self._color_space_field, [fc_int_to_color_space_weight, fc_int_to_color_space_projection])

        color_space_to_fc_cos_projection = DynamicField.Projection(color_space_field_dimensionality, self._find_color.get_cos_field().get_dimensionality(), set([2]), [0])
        color_space_to_fc_cos_weight = DynamicField.Weight(8.0)
        DynamicField.connect(self._color_space_field, self._find_color.get_cos_field(), [color_space_to_fc_cos_projection, color_space_to_fc_cos_weight])

        # create "camera" field
        self._camera_field = CameraField.GaussCameraField()
        self._camera_field.set_name("camera_field")
        self._camera_field_sizes = self._camera_field.get_output_dimension_sizes()

        camera_to_color_space_weight = DynamicField.Weight(4.0)
        DynamicField.connect(self._camera_field, self._color_space_field, [camera_to_color_space_weight])

        # create "spatial target location" field
        spatial_target_field_dimensionality = 2
        spatial_target_kernel = Kernel.GaussKernel(spatial_target_field_dimensionality)
        spatial_target_kernel = Kernel.GaussKernel(spatial_target_field_dimensionality)
        spatial_target_kernel.add_mode(5.0, [1.0] * spatial_target_field_dimensionality, [0.0] * spatial_target_field_dimensionality)
        spatial_target_kernel.calculate()

        spatial_target_field_sizes = move_ee_field_sizes
        self._spatial_target_field = DynamicField.DynamicField([[spatial_target_field_sizes[0]], [spatial_target_field_sizes[1]]], [], spatial_target_kernel)
        self._spatial_target_field.set_global_inhibition(400.0)
        self._spatial_target_field.set_name("spatial_target_field")

        color_space_to_spatial_target_projection = DynamicField.Projection(color_space_field_dimensionality, spatial_target_field_dimensionality, set([0, 1]), [0, 1])
        color_space_to_spatial_target_weight = DynamicField.Weight(10.0)
        DynamicField.connect(self._color_space_field, self._spatial_target_field, [color_space_to_spatial_target_projection, color_space_to_spatial_target_weight])

        spatial_target_to_move_ee_int_weight = DynamicField.Weight(5.0)
        DynamicField.connect(self._spatial_target_field, self._move_ee.get_intention_field(), [spatial_target_to_move_ee_int_weight])

        # create perception field in end effector space
        perception_ee_field_dimensionality = 2
        perception_ee_kernel = Kernel.GaussKernel(perception_ee_field_dimensionality)
        perception_ee_kernel = Kernel.GaussKernel(perception_ee_field_dimensionality)
        perception_ee_kernel.add_mode(15.0, [1.0] * perception_ee_field_dimensionality, [0.0] * perception_ee_field_dimensionality)
        perception_ee_kernel.calculate()

        perception_ee_field_sizes = move_ee_field_sizes
        self._perception_ee_field = DynamicField.DynamicField([[perception_ee_field_sizes[0]], [perception_ee_field_sizes[1]]], [], perception_ee_kernel)
        self._perception_ee_field.set_global_inhibition(400.0)
        self._perception_ee_field.set_name("perception_ee_field")

        perception_ee_to_move_ee_cos_weight = DynamicField.Weight(3.5)
        DynamicField.connect(self._perception_ee_field, self._move_ee.get_cos_field(), [perception_ee_to_move_ee_cos_weight])

        # create end effector control connectable
        self._end_effector_control = EndEffectorControl.EndEffectorControl(move_ee_field_sizes, head_speed_fraction = 0.3)
        DynamicField.connect(self._move_ee.get_intention_field(), self._end_effector_control)
        #############################################################

    def _step_architecture(self):
        self._task_node.step()
        self._camera_field.step()
        self._find_color.step()
        self._perception_ee_field.step()
        self._color_space_field.step()
        self._spatial_target_field.step()
        self._gripper_intention_field.step()
        self._gripper_cos_field.step()
        self._gripper_open.step()
        self._gripper_close.step()
        self._gripper_open_precondition_node.step()
        self._move_ee.step()
        self._end_effector_control.step()
        self._gripper_close_precondition_node.step()

    def onTimer(self, *args):
        self._step_architecture()

        self.spectrogram_plotdata.set_data('imagedata', self._camera_field.get_activation().max(2))
        self.spectrogram_plot.request_redraw()
        return

#============================================================================
# Attributes to use for the plot view.
size = (500,500)
title = "Nao bla"

#============================================================================
# Demo class that is used by the demo.py application.
#============================================================================

class DemoHandler(Handler):

    def closed(self, info, is_ok):
        """ Handles a dialog-based user interface being closed by the user.
        Overridden here to stop the timer once the window is destroyed.
        """

        info.object.timer.Stop()
        return

class Demo(HasTraits):

    plot = Instance(Component)

    controller = Instance(TimerController, ())

    timer = Instance(Timer)

    traits_view = View(
                    Group(
                        Item('plot', editor=ComponentEditor(size=size),
                             show_label=False),
                        orientation = "vertical"),
                    resizable=True, title=title,
                    width=size[0], height=size[1],
                    handler=DemoHandler
                    )

    def __init__(self, **traits):
        super(Demo, self).__init__(**traits)
        self.plot = _create_plot_component(self.controller)

    def edit_traits(self, *args, **kws):
        # Start up the timer! We should do this only when the demo actually
        # starts and not when the demo object is created.
        self.timer = Timer(20, self.controller.onTimer)
        return super(Demo, self).edit_traits(*args, **kws)

    def configure_traits(self, *args, **kws):
        # Start up the timer! We should do this only when the demo actually
        # starts and not when the demo object is created.
        self.timer = Timer(20, self.controller.onTimer)
        return super(Demo, self).configure_traits(*args, **kws)

popup = Demo()

#============================================================================
# Stand-alone frame to display the plot.
#============================================================================

from enthought.etsconfig.api import ETSConfig
#import PyQt4.Qt as Qt
import PyQt4.QtCore as QtCore
import PyQt4.QtGui as QtGui
#import sys



class PlotFrame(DemoFrame):
    def _create_window(self):
        self.controller = TimerController()
        container = _create_plot_component(self.controller)

        # start a continuous timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.controller.onTimer)
        self.timer.start(20)

        return Window(self, -1, component=container)

    def closeEvent(self, event):
        # stop the timer
        if getattr(self, "timer", None):
            self.timer.stop()
        return super(PlotFrame, self).closeEvent(event)

if __name__ == "__main__":
    demo_main(PlotFrame, size=size, title=title)
