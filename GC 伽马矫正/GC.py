import os
import numpy as np
import cv2
import natsort
#from skimage import exposure

def RecoverGC(sceneRadiance):
    sceneRadiance = sceneRadiance/255.0
    # clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(2, 2))
    for i in range(3):
        sceneRadiance[:, :, i] =  np.power(sceneRadiance[:, :, i] / float(np.max(sceneRadiance[:, :, i])), 0.7)
    sceneRadiance = np.clip(sceneRadiance*255, 0, 255)
    sceneRadiance = np.uint8(sceneRadiance)
    return sceneRadiance

np.seterr(over='ignore')
if __name__ == '__main__':
    pass
    print('********    file   ********')
    img = cv2.imread('3.jpg' ,cv2.IMREAD_UNCHANGED)
    sceneRadiance = RecoverGC(img)
    print('sceneRadiance',sceneRadiance)
    cv2.imwrite('GC3.jpg', sceneRadiance)
