import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure as ex
import imageio
import math
import threading as th
import time
import queue
import sys

def GaussianPyramid(img,level):
    g=img.copy()
    gp=[g]
    for i in range(level):
        g=cv2.pyrDown(g)
        gp.append(g)
    return gp
    
def LaplacianPyramid(img,level):
    l=img.copy()
    gp=GaussianPyramid(img,level)
    lp=[gp[level]]
    for i in range(level,0,-1):
        size=(gp[i-1].shape[1],gp[i-1].shape[0])
        ge=cv2.pyrUp(gp[i],dstsize=size)
        l=cv2.subtract(gp[i-1],ge)
        lp.append(l)
    lp.reverse()
    return lp

def PyramidReconstruct(lapl_pyr):
    output = None
    output = np.zeros((lapl_pyr[0].shape[0],lapl_pyr[0].shape[1]), dtype=np.float64)
    for i in range(len(lapl_pyr)-1,0,-1):
        lap = cv2.pyrUp(lapl_pyr[i])
        lapb = lapl_pyr[i-1]
        if lap.shape[0] > lapb.shape[0]:
            lap = np.delete(lap,(-1),axis=0)
        if lap.shape[1] > lapb.shape[1]:
            lap = np.delete(lap,(-1),axis=1)
        tmp = lap + lapb
        lapl_pyr.pop()
        lapl_pyr.pop()
        lapl_pyr.append(tmp)
        output = tmp
    return output
  
def Exposedness(img):
    sigma=0.25
    average=0.5
    row=img.shape[0]
    col=img.shape[1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.normalize(gray, dst=gray, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    res=np.zeros((row,col), np.float32)
    for i in range(row):
        for j in range(col):
            res[i,j]=math.exp(-1.0*math.pow(gray[i,j]-average,2.0)/(2*math.pow(sigma,2.0)))
    res=(res*255)
    res = cv2.convertScaleAbs(res)
    return res       


def Fusion(w1,w2,img1,img2):
    a = time.time()
    
    level=5
    weight1=GaussianPyramid(w1,level)
    weight2=GaussianPyramid(w2,level)
    b1,g1,r1=cv2.split(img1)
    b_pyr1=LaplacianPyramid(b1,level)
    g_pyr1=LaplacianPyramid(g1,level)
    r_pyr1=LaplacianPyramid(r1,level)
    b2,g2,r2=cv2.split(img2)
    b_pyr2=LaplacianPyramid(b2,level)
    g_pyr2=LaplacianPyramid(g2,level)
    r_pyr2=LaplacianPyramid(r2,level)
    b_pyr=[]
    g_pyr=[]
    r_pyr=[]
    for i in range(level):
        b_pyr.append(cv2.add(cv2.multiply(weight1[i],b_pyr1[i]),cv2.multiply(weight2[i],b_pyr2[i])))
        g_pyr.append(cv2.add(cv2.multiply(weight1[i],g_pyr1[i]),cv2.multiply(weight2[i],g_pyr2[i])))
        r_pyr.append(cv2.add(cv2.multiply(weight1[i],r_pyr1[i]),cv2.multiply(weight2[i],r_pyr2[i])))
    b_channel=PyramidReconstruct(b_pyr)
    g_channel=PyramidReconstruct(g_pyr)
    r_channel=PyramidReconstruct(r_pyr)
    out_img=cv2.merge((b_channel,g_channel,r_channel))
    
    print("Normal done in ", time.time()-a)
    return out_img
    
def ApplyGuass_ll(w1, w2, level):
    que = queue.Queue()
    #weight1=GaussianPyramid(w1,level)
    #weight2=GaussianPyramid(w2,level)
    t1 = th.Thread(target=lambda q, arg1,arg2: q.put(GaussianPyramid(arg1,arg2)), args=(que, w1, level))
    t1.start()
    t1.join()
    t2 = th.Thread(target=lambda q, arg1,arg2: q.put(GaussianPyramid(arg1,arg2)), args=(que, w2, level))
    t2.start()
    t2.join()
    weight1 = que.get()
    weight2 = que.get()
    arr = [weight1, weight2]
    return arr

def ApllyLaplac_ll(b1, g1, r1, b2, g2, r2, level):
    que = queue.Queue()
    #b_pyr1=LaplacianPyramid(b1,level)
    #g_pyr1=LaplacianPyramid(g1,level)
    #r_pyr1=LaplacianPyramid(r1,level)
    t3 = th.Thread(target=lambda q, arg1,arg2: q.put(LaplacianPyramid(arg1,arg2)), args=(que, b1, level))
    t3.start()
    t3.join()
    t4 = th.Thread(target=lambda q, arg1,arg2: q.put(LaplacianPyramid(arg1,arg2)), args=(que, g1, level))
    t4.start()
    t4.join()
    t5 = th.Thread(target=lambda q, arg1,arg2: q.put(LaplacianPyramid(arg1,arg2)), args=(que, r1, level))
    t5.start()
    t5.join()
    #b_pyr2=LaplacianPyramid(b2,level)
    #g_pyr2=LaplacianPyramid(g2,level)
    #r_pyr2=LaplacianPyramid(r2,level)
    t6 = th.Thread(target=lambda q, arg1,arg2: q.put(LaplacianPyramid(arg1,arg2)), args=(que, b2, level))
    t6.start()
    t6.join()
    t7 = th.Thread(target=lambda q, arg1,arg2: q.put(LaplacianPyramid(arg1,arg2)), args=(que, g2, level))
    t7.start()
    t7.join()
    t8 = th.Thread(target=lambda q, arg1,arg2: q.put(LaplacianPyramid(arg1,arg2)), args=(que, r2, level))
    t8.start()
    #join all
    t8.join()
    #getting result
    arr = []
    for i in range(6):
        arr.append(que.get())
    return arr

def ApplyFusion_ll(b_pyr, g_pyr, r_pyr):
    que = queue.Queue()
    #b_channel=PyramidReconstruct(b_pyr)
    #g_channel=PyramidReconstruct(g_pyr)
    #r_channel=PyramidReconstruct(r_pyr)
    x1 = th.Thread(target=lambda q, arg1: q.put(PyramidReconstruct(arg1)), args=(que, b_pyr))
    x1.start()
    x1.join()
    x2 = th.Thread(target=lambda q, arg1: q.put(PyramidReconstruct(arg1)), args=(que, g_pyr))
    x2.start()
    x2.join()
    x3 = th.Thread(target=lambda q, arg1: q.put(PyramidReconstruct(arg1)), args=(que, r_pyr))
    x3.start()
    x3.join()
    #getting result
    arr = []
    for i in range(3):
        arr.append(que.get())
    return arr
    

def Fusion_Pll(w1,w2,img1,img2):
    a = time.time()
    level=5
    #GuassianPyramid
    weight = ApplyGuass_ll(w1, w2, level)
    weight1 = weight[0]
    weight2 = weight[1]
    b1,g1,r1=cv2.split(img1)
    b2,g2,r2=cv2.split(img2)
    #LaplacianPyramid
    pyr = ApllyLaplac_ll(b1, g1, r1, b2, g2, r2, level)
    b_pyr1 = pyr[0]
    g_pyr1 = pyr[1]
    r_pyr1 = pyr[2]
    b_pyr2 = pyr[3]
    g_pyr2 = pyr[4]
    r_pyr2 = pyr[5]
    #prefusioncode
    b_pyr=[]
    g_pyr=[]
    r_pyr=[]
    for i in range(level):
        b_pyr.append(cv2.add(cv2.multiply(weight1[i],b_pyr1[i]),cv2.multiply(weight2[i],b_pyr2[i])))
        g_pyr.append(cv2.add(cv2.multiply(weight1[i],g_pyr1[i]),cv2.multiply(weight2[i],g_pyr2[i])))
        r_pyr.append(cv2.add(cv2.multiply(weight1[i],r_pyr1[i]),cv2.multiply(weight2[i],r_pyr2[i])))
        
    channel = ApplyFusion_ll(b_pyr, g_pyr, r_pyr)
    b_channel = channel[0]
    g_channel = channel[1]
    r_channel = channel[2]
    
    out_img=cv2.merge((b_channel,g_channel,r_channel))
    print("Parallel done in ", time.time()-a)
    return out_img


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
    nimg[0] = np.minimum(nimg[0] * (brightest/float(nimg[0].max())),255)
    nimg[1] = np.minimum(nimg[1] * (brightest/float(nimg[1].max())),255)
    nimg[2] = np.minimum(nimg[2] * (brightest/float(nimg[2].max())),255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)


def calculatePx(pmg, brightest, i):
    pmg[i] = np.minimum(pmg[i]*(brightest/float(pmg[i].max())), 255)

def white_balance_ll(nimg):
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
    
    x1 = th.Thread(target=calculatePx, args=(nimg, brightest, 0,))
    x2 = th.Thread(target=calculatePx, args=(nimg, brightest, 1,))
    x3 = th.Thread(target=calculatePx, args=(nimg, brightest, 2,))
    x1.start()
    x2.start()
    x3.start()
    x1.join()
    x2.join()
    x3.join()
    
    return nimg.transpose(1, 2, 0).astype(np.uint8)
    
    
def color_balance(img,percent):
    if percent<=0:
        percent=5 # taken as an average of (1-10).
        
    rows=img.shape[0]
    cols=img.shape[1]
    no_of_chanl=img.shape[2] # knowing the no. of channels in the present image
    
    halfpercent = percent/200.0 # halving the given percentage based on the given research paper
    
    channels=[] # list for storing all the present channels of the image separately.
    
    if no_of_chanl==3:
        for i in range(3):
            channels.append(img[:,:,i:i+1]) # add all the present channels of the image to this list separately
    else:
        channels.append(img)
        
    results=[]
    
    for i in range(no_of_chanl):
        #print(channels[i].shape)
        plane=channels[i].reshape(1,rows*cols,1)
        plane.sort()
        lower_value= plane[0][int(plane.shape[1]*halfpercent)][0]
        top_value = plane[0][int(plane.shape[1]*(1-halfpercent))][0]
        
        channel = channels[i]
        
        for p in range(rows):
            for q in range(cols):
                if channel[p][q][0] < lower_value :
                    channel[p][q][0]=lower_value
                if channel[p][q][0] < top_value :
                    channel[p][q][0]=top_value
        
        channel=cv2.normalize(channel,None,0.0,255.0/2,cv2.NORM_MINMAX)
        # convert the image in desired format-converted
        
        results.append(channel)
        
    output_image = np.zeros((rows,cols,3))
    #for x in results:
        #cv2.imshow('image',x)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    output_image=cv2.merge(results)
    return output_image
  
  
def he(img):
    if(len(img.shape)==2):      #gray
        outImg = ex.equalize_hist(img[:,:])*255 
    elif(len(img.shape)==3):    #RGB
        outImg = np.zeros((img.shape[0],img.shape[1],3))
        for channel in range(img.shape[2]):
            outImg[:, :, channel] = ex.equalize_hist(img[:, :, channel])*255

    outImg[outImg>255] = 255
    outImg[outImg<0] = 0
    return outImg.astype(np.uint8)
    

def pcont(outImg , channel, img):
    outImg[:, :, channel] = ex.equalize_hist(img[:, :, channel])*255

def he_ll(img):
    if(len(img.shape)==2):      #gray
        outImg = ex.equalize_hist(img[:,:])*255 
    elif(len(img.shape)==3):    #RGB
        outImg = np.zeros((img.shape[0],img.shape[1],3))
        #for channel in range(img.shape[2]):
         #   sid = th.Thread(target=lambda q, arg1, arg2, arg3: q.put(pcont(arg1, arg2, arg3)), args=(que, outImg, channel, img))
          #  sid.start()
           # sid.join()
        s1 = th.Thread(target=pcont, args=(outImg, 0, img,))
        s1.start()   
        s2 = th.Thread(target=pcont, args=(outImg, 1, img,))
        s2.start()
        pcont(outImg , 2, img)
        s2.join() 
        s1.join()    
        

    outImg[outImg>255] = 255
    outImg[outImg<0] = 0
    return outImg.astype(np.uint8)
    
    
amg = cv2.imread("1.jpg")                   #读取图片
a = time.time()
conimg = he_ll(amg)
print("Time taken ", time.time()-a)

cv2.imwrite("conimg.jpg",conimg)
#cv2.imshow("1",conimg)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

a = time.time()
amg = cv2.imread("1.jpg")
final_image = white_balance(amg)
percent=100.0
i4 = color_balance(final_image,percent)


#img_name = sys.argv[1]
i5 = he(amg)
print("Time taken ", time.time()-a)

cv2.imwrite("i4.jpg",i4)
#cv2.imshow("2",i4)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

cv2.imwrite("i5.jpg",i5)
#cv2.imshow("3",i5)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

def reqFun(img):
    fmg = white_balance_ll(img)
    percent = 100.0
    i4 = color_balance(final_image,percent)
    return i4
    
    
    
a = time.time()
amg = cv2.imread("1.jpg")
que = queue.Queue()
th1 = th.Thread(target=lambda q, arg1: q.put(reqFun(arg1)), args=(que, amg))
th1.start()

hemg = he(amg)
th1.join()
wmg = que.get()
print("Time taken ", time.time()-a)

cv2.imwrite("wmg.jpg",wmg)
#cv2.imshow("4",wmg)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

cv2.imwrite("he.jpg",hemg)
#cv2.imshow("5",hemg)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

img4 = i4 #cv2.resize(i4,(512,512),interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)

cv2.imwrite("Fusion.jpg",Fusion(gray,gray,img4,final_image))
#cv2.imshow("6",Fusion(gray,gray,img4,final_image))
#cv2.waitKey(0)
#cv2.destroyAllWindows()

cv2.imwrite("Fusion_Pll.jpg",Fusion_Pll(gray,gray,img4,final_image))
#cv2.imshow("7",Fusion_Pll(gray,gray,img4,final_image))
#cv2.waitKey(0)
#cv2.destroyAllWindows()

