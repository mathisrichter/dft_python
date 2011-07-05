#!/usr/bin/env python
"""
This plot displays the audio spectrum from the microphone.

Based on updating_plot.py
"""
# Major library imports
import PyQt4.QtCore as QtCore
import PyQt4.QtGui as QtGui
import PyQt4.Qt as Qt
import PyQt4.Qwt5 as Qwt

import sys
import BehavioralOrganization as BehOrg
import DynamicField
import Kernel
import numpy
import math
import math_tools
import CameraField
import HeadSensorField
import HeadControl

from enthought.etsconfig.etsconfig import ETSConfig
ETSConfig.toolkit = "qt4"

# Enthought library imports
from enthought.chaco.default_colormaps import jet
#from enthought.chaco.api import ColorBar, LinearMapper
from enthought.enable.api import Window, Component, ComponentEditor
from enthought.traits.api import HasTraits, Instance, Range
from enthought.traits.ui.api import Item, Group, View, Handler
from enthought.pyface.timer.api import Timer

# Chaco imports
from enthought.chaco.api import Plot, ArrayPlotData, HPlotContainer, VPlotContainer



#============================================================================
# Create the Chaco plot.
#============================================================================

# HasTraits class that supplies the callable for the timer event.
class TimerController(HasTraits):

    def __init__(self):
        self.arch = BehOrg.GraspArchitecture()
        self._time_steps = 0
        self.create_plot_component()

    def get_container(self):
        return self._container

    def create_plot_component(self):
        color_range_max_value = 10


        # gripper right cos field
        x_axis = numpy.array(range(self.arch._gripper_right_cos_field.get_output_dimension_sizes()[0]))
        self._gripper_right_cos_field_plotdata = ArrayPlotData(x = x_axis, y = self.arch._gripper_right_cos_field.get_activation())
        self._gripper_right_cos_field_plot = Plot(self._gripper_right_cos_field_plotdata)
        self._gripper_right_cos_field_plot.title = 'gripper right cos'
        self._gripper_right_cos_field_plot.plot(("x","y"), name='gripper_right_cos', type = "line", color = "blue")
        range_self = self._gripper_right_cos_field_plot.plots['gripper_right_cos'][0].value_mapper.range
        range_self.high = color_range_max_value
        range_self.low = -color_range_max_value

        # gripper left cos field
        x_axis = numpy.array(range(self.arch._gripper_left_cos_field.get_output_dimension_sizes()[0]))
        self._gripper_left_cos_field_plotdata = ArrayPlotData(x = x_axis, y = self.arch._gripper_left_cos_field.get_activation())
        self._gripper_left_cos_field_plot = Plot(self._gripper_left_cos_field_plotdata)
        self._gripper_left_cos_field_plot.title = 'gripper left cos'
        self._gripper_left_cos_field_plot.plot(("x","y"), name='gripper_left_cos', type = "line", color = "blue")
        range_self = self._gripper_left_cos_field_plot.plots['gripper_left_cos'][0].value_mapper.range
        range_self.high = color_range_max_value
        range_self.low = -color_range_max_value

        # find red color intention field
        x_axis = numpy.array(range(self.arch._find_color.get_intention_field().get_output_dimension_sizes()[0]))
        self._find_color_intention_field_plotdata = ArrayPlotData(x = x_axis, y = self.arch._find_color.get_intention_field().get_activation())
        self._find_color_intention_field_plot = Plot(self._find_color_intention_field_plotdata)
        self._find_color_intention_field_plot.title = 'find color int'
        self._find_color_intention_field_plot.plot(("x","y"), name='find_color_int', type = "line", color = "blue")
        range_self = self._find_color_intention_field_plot.plots['find_color_int'][0].value_mapper.range
        range_self.high = color_range_max_value
        range_self.low = -color_range_max_value

        # find green color intention field
        x_axis = numpy.array(range(self.arch._find_color_ee.get_intention_field().get_output_dimension_sizes()[0]))
        self._find_color_ee_intention_field_plotdata = ArrayPlotData(x = x_axis, y = self.arch._find_color_ee.get_intention_field().get_activation())
        self._find_color_ee_intention_field_plot = Plot(self._find_color_ee_intention_field_plotdata)
        self._find_color_ee_intention_field_plot.title = 'find color ee int'
        self._find_color_ee_intention_field_plot.plot(("x","y"), name='find_color_ee_int', type = "line", color = "blue")
        range_self = self._find_color_ee_intention_field_plot.plots['find_color_ee_int'][0].value_mapper.range
        range_self.high = color_range_max_value
        range_self.low = -color_range_max_value

        # camera
        self._camera_field_plotdata = ArrayPlotData()
        self._camera_field_plotdata.set_data('imagedata', self.arch._camera_field.get_activation().max(2).transpose())
        self._camera_field_plot = Plot(self._camera_field_plotdata)
        self._camera_field_plot.title = 'camera'
        self._camera_field_plot.img_plot('imagedata',
                                  name='camera_field',
                                  xbounds=(0, self.arch._camera_field_sizes[0]-1),
                                  ybounds=(0, self.arch._camera_field_sizes[1]-1),
                                  colormap=jet,
                                  )
        range_self = self._camera_field_plot.plots['camera_field'][0].value_mapper.range
        range_self.high = color_range_max_value
        range_self.low = -color_range_max_value

        # color space red
        self._color_space_field_plotdata = ArrayPlotData()
        self._color_space_field_plotdata.set_data('imagedata', self.arch._color_space_field.get_activation().max(1).transpose())
        self._color_space_field_plot = Plot(self._color_space_field_plotdata)
        self._color_space_field_plot.title = 'color space'
        self._color_space_field_plot.img_plot('imagedata',
                                  name='color_space_field',
                                  xbounds=(0, self.arch._color_space_field_sizes[0]-1),
                                  ybounds=(0, self.arch._color_space_field_sizes[2]-1),
                                  colormap=jet,
                                  )
        range_self = self._color_space_field_plot.plots['color_space_field'][0].value_mapper.range
        range_self.high = color_range_max_value
        range_self.low = -color_range_max_value

        # color space green
        self._color_space_ee_field_plotdata = ArrayPlotData()
        self._color_space_ee_field_plotdata.set_data('imagedata', self.arch._color_space_ee_field.get_activation().max(2).transpose())
        self._color_space_ee_field_plot = Plot(self._color_space_ee_field_plotdata)
        self._color_space_ee_field_plot.title = 'color space ee'
        self._color_space_ee_field_plot.img_plot('imagedata',
                                  name='color_space_ee_field',
                                  xbounds=(0, self.arch._color_space_ee_field_sizes[0]-1),
                                  ybounds=(0, self.arch._color_space_ee_field_sizes[1]-1),
                                  colormap=jet,
                                  )
        range_self = self._color_space_ee_field_plot.plots['color_space_ee_field'][0].value_mapper.range
        range_self.high = color_range_max_value
        range_self.low = -color_range_max_value

        # spatial target
        self._spatial_target_field_plotdata = ArrayPlotData()
        self._spatial_target_field_plotdata.set_data('imagedata', self.arch._spatial_target_field.get_activation().transpose())
        self._spatial_target_field_plot = Plot(self._spatial_target_field_plotdata)
        self._spatial_target_field_plot.title = 'spatial target'
        self._spatial_target_field_plot.img_plot('imagedata',
                                  name='spatial_target_field',
                                  xbounds=(0, self.arch._spatial_target_field_sizes[0]-1),
                                  ybounds=(0, self.arch._spatial_target_field_sizes[1]-1),
                                  colormap=jet,
                                  )
        range_self = self._spatial_target_field_plot.plots['spatial_target_field'][0].value_mapper.range
        range_self.high = color_range_max_value
        range_self.low = -color_range_max_value

        # move head intention
        self._move_head_intention_field_plotdata = ArrayPlotData()
        self._move_head_intention_field_plotdata.set_data('imagedata', self.arch._move_head.get_intention_field().get_activation().transpose())
        self._move_head_intention_field_plot = Plot(self._move_head_intention_field_plotdata)
        self._move_head_intention_field_plot.title = 'move head int'
        self._move_head_intention_field_plot.img_plot('imagedata',
                                  name='move_head_intention_field',
                                  xbounds=(0, self.arch._move_head_field_sizes[0]-1),
                                  ybounds=(0, self.arch._move_head_field_sizes[1]-1),
                                  colormap=jet,
                                  )
        range_self = self._move_head_intention_field_plot.plots['move_head_intention_field'][0].value_mapper.range
        range_self.high = color_range_max_value
        range_self.low = -color_range_max_value

        # move head cos
        self._move_head_cos_field_plotdata = ArrayPlotData()
        self._move_head_cos_field_plotdata.set_data('imagedata', self.arch._move_head.get_cos_field().get_activation().transpose())
        self._move_head_cos_field_plot = Plot(self._move_head_cos_field_plotdata)
        self._move_head_cos_field_plot.title = 'move head cos'
        self._move_head_cos_field_plot.img_plot('imagedata',
                                  name='move_head_cos_field',
                                  xbounds=(0, self.arch._move_head_field_sizes[0]-1),
                                  ybounds=(0, self.arch._move_head_field_sizes[1]-1),
                                  colormap=jet,
                                  )
        range_self = self._move_head_cos_field_plot.plots['move_head_cos_field'][0].value_mapper.range
        range_self.high = color_range_max_value
        range_self.low = -color_range_max_value

        # move right arm intention
        self._move_right_arm_intention_field_plotdata = ArrayPlotData()
        self._move_right_arm_intention_field_plotdata.set_data('imagedata', self.arch._move_right_arm_intention_field.get_activation().transpose())
        self._move_right_arm_intention_field_plot = Plot(self._move_right_arm_intention_field_plotdata)
        self._move_right_arm_intention_field_plot.title = 'move right arm int'
        self._move_right_arm_intention_field_plot.img_plot('imagedata',
                                  name='move_right_arm_intention_field',
                                  xbounds=(0, self.arch._move_arm_field_sizes[0]-1),
                                  ybounds=(0, self.arch._move_arm_field_sizes[1]-1),
                                  colormap=jet,
                                  )
        range_self = self._move_right_arm_intention_field_plot.plots['move_right_arm_intention_field'][0].value_mapper.range
        range_self.high = color_range_max_value
        range_self.low = -color_range_max_value

        # move right arm cos
        self._move_right_arm_cos_field_plotdata = ArrayPlotData()
        self._move_right_arm_cos_field_plotdata.set_data('imagedata', self.arch._move_arm_cos_field.get_activation().transpose())
        self._move_right_arm_cos_field_plot = Plot(self._move_right_arm_cos_field_plotdata)
        self._move_right_arm_cos_field_plot.title = 'move right arm cos'
        self._move_right_arm_cos_field_plot.img_plot('imagedata',
                                  name='move_right_arm_cos_field',
                                  xbounds=(0, self.arch._move_arm_field_sizes[0]-1),
                                  ybounds=(0, self.arch._move_arm_field_sizes[1]-1),
                                  colormap=jet,
                                  )
        range_self = self._move_right_arm_cos_field_plot.plots['move_right_arm_cos_field'][0].value_mapper.range
        range_self.high = color_range_max_value
        range_self.low = -color_range_max_value

        # visual servoing right intention
        self._visual_servoing_right_intention_field_plotdata = ArrayPlotData()
        self._visual_servoing_right_intention_field_plotdata.set_data('imagedata', self.arch._visual_servoing_right.get_intention_field().get_activation().transpose())
        self._visual_servoing_right_intention_field_plot = Plot(self._visual_servoing_right_intention_field_plotdata)
        self._visual_servoing_right_intention_field_plot.title = 'visual servoing right int'
        self._visual_servoing_right_intention_field_plot.img_plot('imagedata',
                                  name='visual_servoing_right_intention_field',
                                  xbounds=(0, self.arch._visual_servoing_field_sizes[0]-1),
                                  ybounds=(0, self.arch._visual_servoing_field_sizes[1]-1),
                                  colormap=jet,
                                  )
        range_self = self._visual_servoing_right_intention_field_plot.plots['visual_servoing_right_intention_field'][0].value_mapper.range
        range_self.high = color_range_max_value
        range_self.low = -color_range_max_value

        # visual servoing right cos
        self._visual_servoing_right_cos_field_plotdata = ArrayPlotData()
        self._visual_servoing_right_cos_field_plotdata.set_data('imagedata', self.arch._visual_servoing_right.get_cos_field().get_activation().transpose())
        self._visual_servoing_right_cos_field_plot = Plot(self._visual_servoing_right_cos_field_plotdata)
        self._visual_servoing_right_cos_field_plot.title = 'visual servoing right cos'
        self._visual_servoing_right_cos_field_plot.img_plot('imagedata',
                                  name='visual_servoing_right_cos_field',
                                  xbounds=(0, self.arch._visual_servoing_field_sizes[0]-1),
                                  ybounds=(0, self.arch._visual_servoing_field_sizes[1]-1),
                                  colormap=jet,
                                  )
        range_self = self._visual_servoing_right_cos_field_plot.plots['visual_servoing_right_cos_field'][0].value_mapper.range
        range_self.high = color_range_max_value
        range_self.low = -color_range_max_value



        self._container = VPlotContainer()
        self._hcontainer_top = HPlotContainer()
        self._hcontainer_bottom = HPlotContainer()
        self._hcontainer_bottom.add(self._camera_field_plot)
        self._hcontainer_bottom.add(self._color_space_field_plot)
        self._hcontainer_bottom.add(self._spatial_target_field_plot)
        self._hcontainer_bottom.add(self._move_head_intention_field_plot)
        self._hcontainer_bottom.add(self._move_right_arm_intention_field_plot)
#        self._hcontainer_bottom.add(self._find_color_intention_field_plot)
#        self._hcontainer_bottom.add(self._gripper_right_intention_field_plot)

        self._hcontainer_top.add(self._color_space_ee_field_plot)
        self._hcontainer_top.add(self._visual_servoing_right_intention_field_plot)
        self._hcontainer_top.add(self._visual_servoing_right_cos_field_plot)
        self._hcontainer_top.add(self._move_head_cos_field_plot)
        self._hcontainer_top.add(self._move_right_arm_cos_field_plot)
#        self._hcontainer_top.add(self._gripper_right_cos_field_plot)

        self._container.add(self._hcontainer_bottom)
        self._container.add(self._hcontainer_top)

    def onTimer(self, *args):
        self.arch.step()

        self._camera_field_plotdata.set_data('imagedata', self.arch._camera_field.get_activation().max(2).transpose())
        self._color_space_field_plotdata.set_data('imagedata', self.arch._color_space_field.get_activation().max(1).transpose())
        self._color_space_ee_field_plotdata.set_data('imagedata', self.arch._color_space_ee_field.get_activation().max(2).transpose())
        self._spatial_target_field_plotdata.set_data('imagedata', self.arch._spatial_target_field.get_activation().transpose())
        self._move_head_intention_field_plotdata.set_data('imagedata', self.arch._move_head.get_intention_field().get_activation().transpose())
        self._move_head_cos_field_plotdata.set_data('imagedata', self.arch._move_head.get_cos_field().get_activation().transpose())
        self._visual_servoing_right_intention_field_plotdata.set_data('imagedata', self.arch._visual_servoing_right.get_intention_field().get_activation().transpose())
        self._visual_servoing_right_cos_field_plotdata.set_data('imagedata', self.arch._visual_servoing_right.get_cos_field().get_activation().transpose())
        self._move_right_arm_intention_field_plotdata.set_data('imagedata', self.arch._move_right_arm_intention_field.get_activation().transpose())
        self._move_right_arm_cos_field_plotdata.set_data('imagedata', self.arch._move_arm_cos_field.get_activation().transpose())
#        self._gripper_right_intention_field_plotdata.set_data('imagedata', self.arch._gripper_right_intention_field.get_activation().transpose())
#        self._gripper_right_cos_field_plotdata.set_data('imagedata', self.arch._gripper_right_cos_field.get_activation().transpose())
#        self._find_color_intention_field_plotdata.set_data('y', self.arch._find_color.get_intention_field().get_activation())
#        self._find_color_ee_intention_field_plotdata.set_data('y', self.arch._find_color_ee.get_intention_field().get_activation())

        self._camera_field_plot.request_redraw()
        self._color_space_field_plot.request_redraw()
        self._color_space_ee_field_plot.request_redraw()
        self._spatial_target_field_plot.request_redraw()
        self._move_head_intention_field_plot.request_redraw()
        self._move_head_cos_field_plot.request_redraw()
        self._visual_servoing_right_intention_field_plot.request_redraw()
        self._visual_servoing_right_cos_field_plot.request_redraw()
        self._move_right_arm_intention_field_plot.request_redraw()
        self._move_right_arm_cos_field_plot.request_redraw()
#        self._gripper_right_cos_field_plot.request_redraw()
#        self._gripper_right_intention_field_plot.request_redraw()

        return




class NaoGuiWidget(QtGui.QWidget):
    def __init__(self, parent = None):
        QtGui.QWidget.__init__(self, parent)

        self.controller = TimerController()
        self.container = self.controller.get_container()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.controller.onTimer)
        self.timer.start(20)
        
        self.chaco_window = Window(self, -1, component=self.container)
        self.field_control_widget = FieldControlWidget(self.controller.arch.fields, self)

        # field control selector
        self.label_field_selector = QtGui.QLabel("controlled field", self)
        self.label_field_selector.setAlignment(Qt.Qt.AlignRight|Qt.Qt.AlignTrailing|Qt.Qt.AlignVCenter)
        self.label_field_selector.setObjectName("label_field_selector")
        self.combo_box_field_selector = QtGui.QComboBox(self)

        for field in self.controller.arch.fields:
            self.combo_box_field_selector.addItem(field.get_name())
        self.change_current_field(0)

        self.connect(self.combo_box_field_selector, Qt.SIGNAL('currentIndexChanged(int)'), self.change_current_field)
        self.connect(self.field_control_widget.line_edit_name, Qt.SIGNAL('editingFinished()'), self.update_combo_box_selector)
        field_control_hbox = QtGui.QHBoxLayout()
        field_control_hbox.addWidget(self.label_field_selector)
        field_control_hbox.addWidget(self.combo_box_field_selector)

        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.chaco_window.control)
        vbox.addLayout(field_control_hbox)
        vbox.addWidget(self.field_control_widget)

        self.setLayout(vbox)

    def update_combo_box_selector(self):
        current_field_index = self.combo_box_field_selector.currentIndex()
        new_field_name = self.field_control_widget.line_edit_name.text()
        self.combo_box_field_selector.setItemText(current_field_index, new_field_name)

    def change_current_field(self, field_index):
        field = self.controller.arch.fields[field_index]
        self.field_control_widget.set_field(field)


class NaoGui(QtGui.QMainWindow):
    def __init__(self, parent = None):
        QtGui.QMainWindow.__init__(self, parent)
        self.setWindowTitle('Nao grasping GUI')
        self.resize(1200, 600)

        nao_gui_widget = NaoGuiWidget()
        self.setCentralWidget(nao_gui_widget)



class FieldControlWidget(QtGui.QWidget):
    def __init__(self, fields, parent = None):
        QtGui.QWidget.__init__(self, parent)
        
        self.fields = fields
        self.current_field = None
        self.setup_controls()
        

    def setup_controls(self):
        self.setWindowTitle('Field dialog')
        self.setObjectName("FieldDialog")
        self.setGeometry(300, 300, 250, 150)

        # name controls
        self.label_name = QtGui.QLabel("name", self)
        self.label_name.setAlignment(Qt.Qt.AlignRight|Qt.Qt.AlignTrailing|Qt.Qt.AlignVCenter)
        self.label_name.setObjectName("label_name")
        
        self.line_edit_name = QtGui.QLineEdit(self)
        self.line_edit_name.setObjectName("line_edit_name")
        self.connect(self.line_edit_name, Qt.SIGNAL('editingFinished()'), self.set_field_name)


        # global inhibition controls
        self.label_global_inhibition = QtGui.QLabel("global inhibition", self)
        self.label_global_inhibition.setAlignment(Qt.Qt.AlignRight|Qt.Qt.AlignTrailing|Qt.Qt.AlignVCenter)
        self.label_global_inhibition.setObjectName("label_global_inhibition")

        self.slider_global_inhibition = Qwt.QwtSlider(self)
        self.slider_global_inhibition.setRange(0.0, 1000.0, 10.0)
        self.slider_global_inhibition.setOrientation(Qt.Qt.Horizontal)
        self.slider_global_inhibition.setBgStyle(Qwt.QwtSlider.BgSlot)
        self.slider_global_inhibition.setObjectName("slider_global_inhibition")
        self.connect(self.slider_global_inhibition, Qt.SIGNAL('valueChanged(double)'), self.set_field_global_inhibition)

        self.value_global_inhibition = QtGui.QLabel(self)
        self.value_global_inhibition.setObjectName("value_global_inhibition")


        # resting level controls
        self.label_resting_level = QtGui.QLabel("resting level", self)
        self.label_resting_level.setAlignment(Qt.Qt.AlignRight|Qt.Qt.AlignTrailing|Qt.Qt.AlignVCenter)
        self.label_resting_level.setObjectName("label_resting_level")

        self.slider_resting_level = Qwt.QwtSlider(self)
        self.slider_resting_level.setRange(-10.0, 0.0, 0.1)
        self.slider_resting_level.setOrientation(Qt.Qt.Horizontal)
        self.slider_resting_level.setBgStyle(Qwt.QwtSlider.BgSlot)
        self.slider_resting_level.setObjectName("slider_resting_level")
        self.connect(self.slider_resting_level, Qt.SIGNAL('valueChanged(double)'), self.set_field_resting_level)

        self.value_resting_level = QtGui.QLabel(self)
        self.value_resting_level.setObjectName("value_resting_level")


        # kernel amplitude controls
        self.label_kernel_amplitude = QtGui.QLabel("kernel amplitude", self)
        self.label_kernel_amplitude.setAlignment(Qt.Qt.AlignRight|Qt.Qt.AlignTrailing|Qt.Qt.AlignVCenter)
        self.label_kernel_amplitude.setObjectName("label_kernel_amplitude")

        self.slider_kernel_amplitude = Qwt.QwtSlider(self)
        self.slider_kernel_amplitude.setRange(0.0, 100.0, 0.1)
        self.slider_kernel_amplitude.setOrientation(Qt.Qt.Horizontal)
        self.slider_kernel_amplitude.setBgStyle(Qwt.QwtSlider.BgSlot)
        self.slider_kernel_amplitude.setObjectName("slider_kernel_amplitude")
        self.connect(self.slider_kernel_amplitude, Qt.SIGNAL('valueChanged(double)'), self.set_field_kernel_amplitude)

        self.value_kernel_amplitude = QtGui.QLabel(self)
        self.value_kernel_amplitude.setObjectName("value_kernel_amplitude")


        # kernel width controls
        self.label_kernel_width = QtGui.QLabel("kernel width", self)
        self.label_kernel_width.setAlignment(Qt.Qt.AlignRight|Qt.Qt.AlignTrailing|Qt.Qt.AlignVCenter)
        self.label_kernel_width.setObjectName("label_kernel_width")

        self.slider_kernel_width = Qwt.QwtSlider(self)
        self.slider_kernel_width.setRange(0.0, 20.0, 0.1)
        self.slider_kernel_width.setOrientation(Qt.Qt.Horizontal)
        self.slider_kernel_width.setBgStyle(Qwt.QwtSlider.BgSlot)
        self.slider_kernel_width.setObjectName("slider_kernel_width")
        self.connect(self.slider_kernel_width, Qt.SIGNAL('valueChanged(double)'), self.set_field_kernel_width)

        self.value_kernel_width = QtGui.QLabel(self)
        self.value_kernel_width.setObjectName("value_kernel_width")


        # noise controls
        self.label_noise = QtGui.QLabel("noise", self)
        self.label_noise.setAlignment(Qt.Qt.AlignRight|Qt.Qt.AlignTrailing|Qt.Qt.AlignVCenter)
        self.label_noise.setObjectName("label_noise")

        self.slider_noise = Qwt.QwtSlider(self)
        self.slider_noise.setRange(0.0, 1.0, 0.01)
        self.slider_noise.setOrientation(Qt.Qt.Horizontal)
        self.slider_noise.setBgStyle(Qwt.QwtSlider.BgSlot)
        self.slider_noise.setObjectName("slider_noise")
        self.connect(self.slider_noise, Qt.SIGNAL('valueChanged(double)'), self.set_field_noise)

        self.value_noise = QtGui.QLabel(self)
        self.value_noise.setObjectName("value_noise")


        # boost controls
        self.label_boost = QtGui.QLabel("boost", self)
        self.label_boost.setAlignment(Qt.Qt.AlignRight|Qt.Qt.AlignTrailing|Qt.Qt.AlignVCenter)
        self.label_boost.setObjectName("label_boost")

        self.slider_boost = Qwt.QwtSlider(self)
        self.slider_boost.setRange(0.0, 20.0, 0.1)
        self.slider_boost.setOrientation(Qt.Qt.Horizontal)
        self.slider_boost.setBgStyle(Qwt.QwtSlider.BgSlot)
        self.slider_boost.setObjectName("slider_boost")
        self.connect(self.slider_boost, Qt.SIGNAL('valueChanged(double)'), self.set_field_boost)

        self.value_boost = QtGui.QLabel(self)
        self.value_boost.setObjectName("value_boost")


        # sigmoid steepness controls
        self.label_steepness = QtGui.QLabel("steepness (sigmoid)", self)
        self.label_steepness.setAlignment(Qt.Qt.AlignRight|Qt.Qt.AlignTrailing|Qt.Qt.AlignVCenter)
        self.label_steepness.setObjectName("label_steepness")

        self.slider_steepness = Qwt.QwtSlider(self)
        self.slider_steepness.setRange(0.0, 100.0, 0.1)
        self.slider_steepness.setOrientation(Qt.Qt.Horizontal)
        self.slider_steepness.setBgStyle(Qwt.QwtSlider.BgSlot)
        self.slider_steepness.setObjectName("slider_steepness")
        self.connect(self.slider_steepness, Qt.SIGNAL('valueChanged(double)'), self.set_field_sigmoid_steepness)

        self.value_steepness = QtGui.QLabel(self)
        self.value_steepness.setObjectName("value_steepness")


        # sigmoid shift controls
        self.label_shift = QtGui.QLabel("shift (sigmoid)", self)
        self.label_shift.setAlignment(Qt.Qt.AlignRight|Qt.Qt.AlignTrailing|Qt.Qt.AlignVCenter)
        self.label_shift.setObjectName("label_shift")

        self.slider_shift = Qwt.QwtSlider(self)
        self.slider_shift.setRange(0.0, 10.0, 0.1)
        self.slider_shift.setOrientation(Qt.Qt.Horizontal)
        self.slider_shift.setBgStyle(Qwt.QwtSlider.BgSlot)
        self.slider_shift.setObjectName("slider_shift")
        self.connect(self.slider_shift, Qt.SIGNAL('valueChanged(double)'), self.set_field_sigmoid_shift)

        self.value_shift = QtGui.QLabel(self)
        self.value_shift.setObjectName("value_shift")


        # relaxation time controls
        self.label_relaxation_time = QtGui.QLabel("relaxation time", self)
        self.label_relaxation_time.setAlignment(Qt.Qt.AlignRight|Qt.Qt.AlignTrailing|Qt.Qt.AlignVCenter)
        self.label_relaxation_time.setObjectName("label_relaxation_time")

        self.slider_relaxation_time = Qwt.QwtSlider(self)
        self.slider_relaxation_time.setRange(1.0, 100.0, 1.0)
        self.slider_relaxation_time.setOrientation(Qt.Qt.Horizontal)
        self.slider_relaxation_time.setBgStyle(Qwt.QwtSlider.BgSlot)
        self.slider_relaxation_time.setObjectName("slider_relaxation_time")
        self.connect(self.slider_relaxation_time, Qt.SIGNAL('valueChanged(double)'), self.set_field_relaxation_time)

        self.value_relaxation_time = QtGui.QLabel(self)
        self.value_relaxation_time.setObjectName("value_relaxation_time")


        # layout
        field_control_grid_layout = QtGui.QGridLayout()

        name_widgets = [self.label_name, self.line_edit_name]
        field_control_grid_layout.addWidget(name_widgets[0], 0, 0)
        field_control_grid_layout.addWidget(name_widgets[1], 0, 1)
        
        global_inhibition_widgets = [self.label_global_inhibition, self.slider_global_inhibition, self.value_global_inhibition]
        field_control_grid_layout.addWidget(global_inhibition_widgets[0], 1, 0)
        field_control_grid_layout.addWidget(global_inhibition_widgets[1], 1, 1)
        field_control_grid_layout.addWidget(global_inhibition_widgets[2], 1, 2)

        resting_level_widgets = [self.label_resting_level, self.slider_resting_level, self.value_resting_level]
        field_control_grid_layout.addWidget(resting_level_widgets[0], 2, 0)
        field_control_grid_layout.addWidget(resting_level_widgets[1], 2, 1)
        field_control_grid_layout.addWidget(resting_level_widgets[2], 2, 2)

        kernel_amplitude_widgets = [self.label_kernel_amplitude, self.slider_kernel_amplitude, self.value_kernel_amplitude]
        field_control_grid_layout.addWidget(kernel_amplitude_widgets[0], 3, 0)
        field_control_grid_layout.addWidget(kernel_amplitude_widgets[1], 3, 1)
        field_control_grid_layout.addWidget(kernel_amplitude_widgets[2], 3, 2)

        kernel_width_widgets = [self.label_kernel_width, self.slider_kernel_width, self.value_kernel_width]
        field_control_grid_layout.addWidget(kernel_width_widgets[0], 4, 0)
        field_control_grid_layout.addWidget(kernel_width_widgets[1], 4, 1)
        field_control_grid_layout.addWidget(kernel_width_widgets[2], 4, 2)


        noise_widgets =  [self.label_noise, self.slider_noise, self.value_noise]
        field_control_grid_layout.addWidget(noise_widgets[0], 5, 0)
        field_control_grid_layout.addWidget(noise_widgets[1], 5, 1)
        field_control_grid_layout.addWidget(noise_widgets[2], 5, 2)

        boost_widgets = [self.label_boost, self.slider_boost, self.value_boost]
        field_control_grid_layout.addWidget(boost_widgets[0], 6, 0)
        field_control_grid_layout.addWidget(boost_widgets[1], 6, 1)
        field_control_grid_layout.addWidget(boost_widgets[2], 6, 2)

        steepness_widgets = [self.label_steepness, self.slider_steepness, self.value_steepness]
        field_control_grid_layout.addWidget(steepness_widgets[0], 7, 0)
        field_control_grid_layout.addWidget(steepness_widgets[1], 7, 1)
        field_control_grid_layout.addWidget(steepness_widgets[2], 7, 2)

        shift_widgets = [self.label_shift, self.slider_shift, self.value_shift]
        field_control_grid_layout.addWidget(shift_widgets[0], 8, 0)
        field_control_grid_layout.addWidget(shift_widgets[1], 8, 1)
        field_control_grid_layout.addWidget(shift_widgets[2], 8, 2)

        relaxation_time_widgets = [self.label_relaxation_time, self.slider_relaxation_time, self.value_relaxation_time]
        field_control_grid_layout.addWidget(relaxation_time_widgets[0], 9, 0)
        field_control_grid_layout.addWidget(relaxation_time_widgets[1], 9, 1)
        field_control_grid_layout.addWidget(relaxation_time_widgets[2], 9, 2)

        field_control_group_box = QtGui.QGroupBox("Field parameters")
        field_control_group_box.setLayout(field_control_grid_layout)
        
        self.hbox = QtGui.QHBoxLayout()
        
        self.hbox.addWidget(field_control_group_box)

        self.setLayout(self.hbox)


    def set_field(self, current_field):
        self.current_field = current_field
    
        self.line_edit_name.setText(self.current_field.get_name())
        
        global_inhibition = self.current_field.get_global_inhibition()
        self.slider_global_inhibition.setValue(global_inhibition)
        self.value_global_inhibition.setNum(global_inhibition)

        resting_level = self.current_field.get_resting_level()
        self.slider_resting_level.setValue(resting_level)
        self.value_resting_level.setNum(resting_level)
        
        kernel_amplitude = self.current_field.get_lateral_interaction_kernel(0).get_amplitude()
        self.slider_kernel_amplitude.setValue(kernel_amplitude)
        self.value_kernel_amplitude.setNum(kernel_amplitude)

        kernel_width = self.current_field.get_lateral_interaction_kernel(0).get_width(0)
        self.slider_kernel_width.setValue(kernel_width)
        self.value_kernel_width.setNum(kernel_width)
        
        noise = self.current_field.get_noise_strength()
        self.slider_noise.setValue(noise)
        self.value_noise.setNum(noise)
        
        boost = self.current_field.get_boost()
        self.slider_boost.setValue(boost)
        self.value_boost.setNum(boost)
        
        sigmoid_steepness = self.current_field.get_sigmoid_steepness()
        self.slider_steepness.setValue(sigmoid_steepness)
        self.value_steepness.setNum(sigmoid_steepness)
        
        sigmoid_shift = self.current_field.get_sigmoid_shift()
        self.slider_shift.setValue(sigmoid_shift)
        self.value_shift.setNum(self.current_field.get_sigmoid_shift())

        relaxation_time = self.current_field.get_relaxation_time()
        self.slider_relaxation_time.setValue(relaxation_time)
        self.value_relaxation_time.setNum(relaxation_time)
        
    def set_field_name(self):
        if self.current_field is not None:
            self.current_field.set_name(str(self.line_edit_name.text()))

    def set_field_boost(self, boost):
        if self.current_field is not None:
            self.current_field.set_boost(boost)
            self.value_resting_level.setNum(boost)

    def set_field_global_inhibition(self, inhibition):
        if self.current_field is not None:
            self.current_field.set_global_inhibition(inhibition)
            self.value_global_inhibition.setNum(inhibition)

    def set_field_resting_level(self, resting_level):
        if self.current_field is not None:
            self.current_field.set_resting_level(resting_level)
            self.value_resting_level.setNum(resting_level)

    def set_field_kernel_amplitude(self, kernel_amplitude):
        if self.current_field is not None:
            self.current_field.get_lateral_interaction_kernel(0).set_amplitude(kernel_amplitude)
            self.value_kernel_amplitude.setNum(kernel_amplitude)

    def set_field_kernel_width(self, kernel_width):
        if self.current_field is not None:
            self.current_field.get_lateral_interaction_kernel(0).set_width(kernel_width, 0)
            self.value_kernel_width.setNum(kernel_width)

    def set_field_noise(self, noise):
        if self.current_field is not None:
            self.current_field.set_noise_strength(noise)
            self.value_noise.setNum(noise)

    def set_field_sigmoid_steepness(self, steepness):
        if self.current_field is not None:
            self.current_field.set_sigmoid_steepness(steepness)
            self.value_steepness.setNum(steepness)

    def set_field_sigmoid_shift(self, shift):
        if self.current_field is not None:
            self.current_field.set_sigmoid_shift(shift)
            self.value_shift.setNum(shift)

    def set_field_boost(self, input):
        if self.current_field is not None:
            self.current_field.set_boost(input)
            self.value_boost.setNum(input)

    def set_field_relaxation_time(self, relaxation_time):
        if self.current_field is not None:
            self.current_field.set_relaxation_time(relaxation_time)
            self.value_relaxation_time.setNum(relaxation_time)

