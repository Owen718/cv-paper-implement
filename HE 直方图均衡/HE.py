import os
import numpy as np
import cv2
import natsort
#import xlwt
#from skimage import exposure

#from sceneRadianceCLAHE import RecoverCLAHE
#from sceneRadianceHE import RecoverHE

def RecoverCLAHE(sceneRadiance):
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4))
    for i in range(3):

        # sceneRadiance[:, :, i] =  cv2.equalizeHist(sceneRadiance[:, :, i])
        sceneRadiance[:, :, i] = clahe.apply((sceneRadiance[:, :, i]))
    return sceneRadiance
    
def RecoverHE(sceneRadiance):
    # clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(2, 2))
    for i in range(3):
        sceneRadiance[:, :, i] =  cv2.equalizeHist(sceneRadiance[:, :, i])
        # sceneRadiance[:, :, i] = clahe.apply((sceneRadiance[:, :, i]))
    return sceneRadiance

np.seterr(over='ignore')
if __name__ == '__main__':
    pass
    print('********    file   ********')
    img = cv2.imread('3.jpg',cv2.IMREAD_UNCHANGED)
    sceneRadiance = RecoverHE(img)
    print('sceneRadiance',sceneRadiance)
    cv2.imwrite('HE3.jpg', sceneRadiance)
