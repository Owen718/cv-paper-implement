import cv2
import numpy as np
import matplotlib.pyplot as plt
#from skimage import exposure as ex
#import imageio
import math
import threading as th
import time
import queue
import sys

def calculatePx(pmg, brightest, i):
    pmg[i] = np.minimum(pmg[i]*(brightest/float(pmg[i].max())), 255)

def white_balance(nimg):
    if nimg.dtype==np.uint8:
        brightest=float(2**8)
    elif nimg.dtype==np.uinAt16:
        brightest=float(2**16)
    elif nimg.dtype==np.uint32:
        brightest=float(2**32)
    else:
        brightest==float(2**8)
    nimg = nimg.transpose(2, 0, 1)
    nimg = nimg.astype(np.int32)
    
    que = queue.Queue()
    
    x1 = th.Thread(target=lambda q, arg1, arg2, arg3: q.put(calculatePx(arg1, arg2, arg3)), args=(que, nimg, brightest, 0))
    x2 = th.Thread(target=lambda q, arg1, arg2, arg3: q.put(calculatePx(arg1, arg2, arg3)), args=(que, nimg, brightest, 1))
    x3 = th.Thread(target=lambda q, arg1, arg2, arg3: q.put(calculatePx(arg1, arg2, arg3)), args=(que, nimg, brightest, 2))
    x1.start()
    x2.start()
    x3.start()
    x1.join()
    x2.join()
    x3.join()
    
    return nimg.transpose(1, 2, 0).astype(np.uint8)
    
    
def white_balance_serial(nimg):
    if nimg.dtype==np.uint8:
        brightest=float(2**8)
    elif nimg.dtype==np.uinAt16:
        brightest=float(2**16)
    elif nimg.dtype==np.uint32:
        brightest=float(2**32)
    else:
        brightest==float(2**8)
    nimg = nimg.transpose(2, 0, 1)
    nimg = nimg.astype(np.int32)
    nimg[0] = np.minimum(nimg[0] * (brightest/float(nimg[0].max())),255)
    nimg[1] = np.minimum(nimg[1] * (brightest/float(nimg[1].max())),255)
    nimg[2] = np.minimum(nimg[2] * (brightest/float(nimg[2].max())),255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)
    
#amg = cv2.imread("3.jpg")
#pll_image = white_balance(amg)

#cv2.imwrite('WB3.jpg', pll_image)

amg = cv2.imread("3.jpg")
pll_image = white_balance_serial(amg)

cv2.imwrite('WB_serial_3.jpg', pll_image)
