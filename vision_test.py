import naoqi
from naoqi import ALProxy
import numpy
import math_tools
import matplotlib.pyplot as plt

def main():
    vision_proxy = ALProxy("ALVideoDevice", "192.168.0.102", 9559)
    gvm_name = "nao vision"
    gvm_name = vision_proxy.subscribe(gvm_name, 0, 12, 15)

    naoimage = vision_proxy.getImageRemote(gvm_name)
    hsv_image = numpy.fromstring(naoimage[6], dtype=numpy.uint8)
    hue = hsv_image[::3].reshape(120,160)
    print(hue.shape)

    sizes = [30,40,15]

    hue = math_tools.linear_interpolation_2d_custom(hue, [sizes[0], sizes[1]])
    hue = numpy.round(hue * ((sizes[2] - 1)/255.)).astype(numpy.int)
    
    plt.imshow(hue)
    plt.show()

    gvm_name = vision_proxy.unsubscribe(gvm_name)

if __name__ == "__main__":
    main()

