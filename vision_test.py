import naoqi
from naoqi import ALProxy
import numpy
import matplotlib.pyplot as plt
import time


vision_proxy = ALProxy("ALVideoDevice", "192.168.0.102", 9559)
gvm_name = "vision stuff"
gvm_name = vision_proxy.subscribe(gvm_name, 0, 12, 15)

time_steps = 100

hue_activation = numpy.zeros((time_steps, 160))

for i in range(time_steps):
    naoimage = vision_proxy.getImageRemote(gvm_name)
    hsv_image = numpy.fromstring(naoimage[6], dtype=numpy.uint8)
    hue = hsv_image[::3].reshape(120,160)
    hue_activation[i] = hue.max(0)
    time.sleep(0.03)

fig = plt.figure(1)
plt.imshow(hue_activation, aspect="auto")
plt.show()

vision_proxy.unsubscribe(gvm_name)

