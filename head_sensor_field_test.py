import HeadSensorField
import time
from naoqi import ALProxy

proxy = ALProxy("ALMotion", "192.168.0.102", 9559)
#proxy.setStiffnesses("Head", 0.0)
proxy.setStiffnesses("RArm", 0.0)

while(True):
#    cam_pos = proxy.getPosition("CameraBottom", 2, True)
    cam_pos = proxy.getPosition("RArm", 2, True)
    print("cam pos: ", cam_pos)

    time.sleep(0.5)

    
