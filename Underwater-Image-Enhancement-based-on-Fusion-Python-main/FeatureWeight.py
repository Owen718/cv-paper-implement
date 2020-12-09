import cv2
import numpy as np
import math


def Saliency(img):
    gfgbr = cv2.GaussianBlur(img,(3, 3), 3)
    LabIm = cv2.cvtColor(gfgbr, cv2.COLOR_BGR2Lab)
    lab = cv2.split(LabIm)
    l = np.float32(lab[0])
    a = np.float32(lab[1])
    b = np.float32(lab[2])
    lm = cv2.mean(l)[0] # cv2.mean(l).val[0]
    am = cv2.mean(a)[0]
    bm = cv2.mean(b)[0]
    sm = np.zeros(l.shape, l[0][1].dtype)
    l = cv2.subtract(l, lm)
    a = cv2.subtract(a, am)
    b = cv2.subtract(b, bm)
    sm = cv2.add(sm, cv2.multiply(l, l)) 
    sm = cv2.add(sm, cv2.multiply(a, a))
    sm = cv2.add(sm, cv2.multiply(b, b))
    return sm


def LaplacianContrast(img):
    # img=cv2.CreateMat(h, w, cv2.CV_32FC3)
    laplacian = cv2.Laplacian(img,5) 
    laplacian = cv2.convertScaleAbs(laplacian)
    return laplacian


def LocalContrast(img):
    h = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0]
    mask = np.ones((len(h),len(h)), img[0][0].dtype)
    for i in range(len(h)):
        for j in range(len(h)):
            mask[i][j]=(h[i] * h[j])
    localContrast = cv2.filter2D(img, 5,kernel=mask) 
    for i in range(len(localContrast)):
        for j in range(len(localContrast[0])):
            if localContrast[i][j] > math.pi / 2.75:
                localContrast[i][j] = math.pi / 2.75
    localContrast = cv2.subtract(img, localContrast)
    return cv2.multiply(localContrast, localContrast)


def Exposedness(img):
    sigma = 0.25
    average = 0.5
    exposedness = np.zeros(img.shape,img[0][0].dtype)
    for i in range(len(img)):
        for j in range(len(img[0])):
            value = math.exp(-1.0 * math.pow(img[i, j] - average, 2.0) / (2 * math.pow(sigma, 2.0)))
            exposedness[i][j] = value
    return exposedness


def LuminanceWeight(img, L):
    bCnl = np.float32(cv2.extractChannel(img, 0))
    gCnl = cv2.extractChannel(img, 1)
    rCnl = cv2.extractChannel(img, 2)
    rCnl = np.float32(rCnl)
    lum = np.zeros(L.shape, L.dtype())
    for i in range(len(L)):
        for j in range(len(L[0])):
            data = math.sqrt((math.pow(bCnl[i][j] / 255.0 - L[i][j], 2.0) + math.pow(
                gCnl[i][j] / 255.0 - L[i][j], 2.0) + math.pow(rCnl[i][j] / 255.0 - L[i][j], 2.0)) / 3)
            lum[i][j] = data
    return lum
