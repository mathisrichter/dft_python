import naoqi
from naoqi import ALProxy
import numpy
import DynamicField
import math_tools

class NaoCameraField(DynamicField.DynamicField):
    "Camera field"

    def __init__(self):
        "Constructor"
        DynamicField.DynamicField.__init__(self, dimension_bounds = [[40],[30],[15]])

        self._vision_proxy = ALProxy("ALVideoDevice", "192.168.0.102", 9559)
        self._gvm_name = "nao vision"
        self._gvm_name = self._vision_proxy.subscribe(self._gvm_name, 0, 12, 15)
        self._name = "nao_camera_field"
        self.start_activation_log()

    def __del__(self):
        self._gvm_name = self._vision_proxy.unsubscribe(self._gvm_name)
        self.stop_activation_log()

    def _step_computation(self):
        naoimage = self._vision_proxy.getImageRemote(self._gvm_name)
        hsv_image = numpy.fromstring(naoimage[6], dtype=numpy.uint8)
        hue = hsv_image[::3].reshape(120,160)

        sizes = self.get_input_dimension_sizes()

        hue = math_tools.linear_interpolation_2d_custom(hue, [sizes[0], sizes[1]])
        hue = numpy.round(hue * ((sizes[2] - 1)/255.)).astype(numpy.int)

        for i in range(sizes[0]):
            for j in range(sizes[1]):
                color = hue[i][j]
                self._activation[i][j] = -5.
                self._activation[i][j][color] = 5.

        self._output_buffer = self.compute_thresholded_activation(self._activation)
        self.write_activation_log()

class GaussCameraField(DynamicField.DynamicField):
    "Camera field"

    def __init__(self):
        "Constructor"
        DynamicField.DynamicField.__init__(self, dimension_bounds = [[40],[30],[15]])

        self._activation += math_tools.gauss_3d([40,30,15], 9.0, [2.0,2.0,2.0], [10,20,0])
        self._output_buffer = self.compute_thresholded_activation(self._activation)

    def _step_computation(self):
        pass

class DummyCameraField(DynamicField.DynamicField):
    "Camera field"

    def __init__(self):
        "Constructor"
        DynamicField.DynamicField.__init__(self, dimension_bounds = [[40],[30],[15]])
        camera_field_file = open("snapshots/camera_field.txt", 'r')
        activation = numpy.fromfile(camera_field_file, sep=', ')
        camera_field_file.close()

        activation = activation.reshape(160,120,50)
        self._activation = math_tools.linear_interpolation_nd(activation, [40, 30, 15])
        self._output_buffer = self.compute_thresholded_activation(self._activation)

    def _step_computation(self):
        pass
