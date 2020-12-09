import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import cv2

def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



img = cv2.imread(r'C:\Users\Owen\Pictures\RGB_3D_test.jpg')
#img  = cv2.resize(img,None,fx=0.1,fy=0.1,interpolation = cv2.INTER_LINEAR)

mpl.rcParams['legend.fontsize'] = 10
 
fig = plt.figure()
ax = fig.gca(projection='3d')


for x in range(img.shape[0]):
        ax.scatter(img[x][:][0],img[x][:][1],img[x][:][2],c=(img[x][:][2]/255,img[x][:][1]/255,img[x][:][0]/255),marker='.')



ax.set_xlabel('R',color = 'r')  #x轴为R
ax.set_ylabel('G',color = 'g')  #y轴为G
ax.set_zlabel('B',color = 'b')  #z轴为B 

cv_show('0.01',img)

plt.show()