import naoqi
from naoqi import ALProxy
import numpy
import DynamicField

class CameraField(DynamicField.DynamicField):
    "Camera field"

    def __init__(self):
        "Constructor"
        DynamicField.DynamicField.__init__(self, dimension_bounds = [[160],[120],[50]])

        self._vision_proxy = ALProxy("ALVideoDevice", "192.168.0.102", 9559)
        self._gvm_name = "nao vision"
        self._gvm_name = self._vision_proxy.subscribe(self._gvm_name, 0, 12, 15)

    def __del__(self):
        self._gvm_name = self._vision_proxy.unsubscribe(self._gvm_name)

    def _step_computation(self):
        naoimage = self._vision_proxy.getImageRemote(self._gvm_name)
        hsv_image = numpy.fromstring(naoimage[6], dtype=numpy.uint8)
        hue = hsv_image[::3].reshape(120,160).transpose()
        hue = numpy.round(hue * (self.get_input_dimension_sizes()[2]/255.)) - 1

        for i in range(self.get_input_dimension_sizes()[0]):
            for j in range(self.get_input_dimension_sizes()[1]):
                color = hue[i][j]
                self._activation[i][j] = -5.
                self._activation[i][j][color] = 5.

        print "stepped camera field"
