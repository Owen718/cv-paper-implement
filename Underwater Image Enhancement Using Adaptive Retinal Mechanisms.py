import cv2
import numpy as np
import math
from skimage import exposure

def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def mat_mean(img,θ,N):  #计算各通道均值，根据θ来判断返回R通道最大值或平均值,目前暂定N为3
    global original_img
    mean_kernel = np.array([
        [1/9,1/9,1/9],
        [1/9,1/9,1/9],
        [1/9,1/9,1/9]
    ])   #均值算子
    #B_mean = cv2.filter2D(img[:,:,0],ddepth=-1,kernel=mean_kernel)  #B通道卷积计算
    #G_mean = cv2.filter2D(img[:,:,1],ddepth=-1,kernel=mean_kernel)  #G通道卷积计算

    B_mean =cv2.boxFilter(img[:,:,0],ddepth=-1,ksize=(N,N),normalize=True)
    G_mean =cv2.boxFilter(img[:,:,1],ddepth=-1,ksize=(N,N),normalize=True)


    #B_mean = cv2.blur(img[:,:,0],ksize=(N,N),anchor=(-1,-1))
    #G_mean = cv2.blur(img[:,:,1],ksize=(N,N),anchor=(-1,-1))
    R = img[:,:,2]
    row,col = R.shape  #row是行，col是列
    max_num  = 0
    mean_num = 0
    original_view = original_img.copy()
    R_mean = np.zeros((row,col),dtype = np.float64)
    for x in range(row-N):  #遍历行
        for y in range(col-N):  #遍历列
            xy_zone = R[x:x+N,y:y+N] 
            if xy_zone.max() > θ:  #若是最大值大于θ
               ## for i in range(xy_zone.shape[0]):
                   # for j in range(xy_zone.shape[1]):
                   #     if xy_zone[i,j]<θ:
                    #       xy_zone[i,j]=θ
                #if  np.mean(xy_zone) < θ:
                #    R_mean[x,y]=θ
                #else:
                #R_mean[x,y]=np.mean(xy_zone)
                #R_mean[x:x+3,y:y+3]=cv2.boxFilter(xy_zone,ddepth=-1,ksize=(N,N),anchor=None,normalize=False)
                #max_num +=1
                R_mean[x,y]=xy_zone.max()
                mean_num += 1
            else:  #若是最大值小于θ
                #R_mean[x,y]=np.mean(xy_zone)  #取均值
                #R_mean[x:x+3,y:y+3]=cv2.boxFilter(xy_zone,ddepth=-1,ksize=(N,N),normalize=False)
                #mean_num += 1
                max_num += 1
                R_mean[x,y]=np.mean(xy_zone)
                #original_view = cv2.circle(original_view,center=(y,x),radius = 1,color = (255,255,255))#绘制区域
                original_view[x,y,:] = [0,0,255]
                
    cv_show('original_view',original_view)
    print('max_num:'+str(max_num))
    print('mean_num:'+str(mean_num))       

    for x in range(row-N,row):  #复制图像边缘
        for y in range(col):
            R_mean[x,y] = R[x,y]

    for x in range(row):  #复制图像边缘
        for y in range(col-N,col):
            R_mean[x,y] = R[x,y]


    return B_mean.astype(np.float64),G_mean.astype(np.float64),R_mean.astype(np.float64)



def L_mean(L,N): #单通道均值函数
    mean_kernel = np.array([
        [1/9,1/9,1/9],
        [1/9,1/9,1/9],
        [1/9,1/9,1/9]
    ]) 
    #L_mean = cv2.filter2D(L,ddepth=-1,kernel=mean_kernel)  #B通道卷积计算
    L_mean = cv2.boxFilter(L,ddepth=-1,ksize=(N,N),normalize=True)
    return L_mean  

def RGB_merge(name,B,G,R):
    img=cv2.merge([B,G,R])
    cv_show(name,img)

def gaussian_2d_funcion(Rx,Ry):  #计算二维高斯函数值
    gaussian_sigma = 1/(2*np.pi*σ*σ) * np.exp(-(Rx*Rx + Ry*Ry)/(2*σ*σ))
    return gaussian_sigma

def generate_gaussian_2d_matrix(shape):  #生成二维高斯函数矩阵
    gaussian_2d_matrix = np.zeros(shape[0],shape[1])  #生成shape[0]行，shape[1]列的高斯滤波矩阵
    for row in range(shape[0]):  #遍历行
        for col in range(shape[1]):  #遍历列
            gaussian_2d_matrix[row,col] = gaussian_2d_funcion(row,col)
    return gaussian_2d_matrix

def gamma_enhance(bgr,c,a):  #s=c*pow(r,a)  #gamma函数
    (b,g,r) = cv2.split(bgr)
    for x in range(bgr.shape[0]):
        for y in range(bgr.shape[1]):
            b[x,y] = c * math.pow(b[x,y]/255,a) * 255
            g[x,y] = c * math.pow(g[x,y]/255,a) * 255
            r[x,y] = c * math.pow(r[x,y]/255,a) * 255

    img = cv2.merge((b,g,r))
    cv2.normalize(img,img,0,255,cv2.NORM_MINMAX)
    m_img=cv2.convertScaleAbs(img)
    return m_img

def max_mat(num,Bp):  #将Bp中所有小于num的值替换为num
    for x in range(Bp.shape[0]):
        for y in range(Bp.shape[1]):
            if Bp[x,y] < num:
                Bp[x,y] = num
    
    return Bp

def zero_to_one(HCF):  #可能出现0的情况，无法作为除数，将0替换为1（0-255，uint8）
    for x in range(HCF.shape[0]):
        for y in range(HCF.shape[1]):
            if HCF[x,y]==0:
                HCF[x,y]=1
    return HCF

#求暗通道：
def getDarkChannel(img, blockSize):
    # 输入检查
    if len(img.shape) == 2:
        pass
    else:
        print("bad image shape, input image must be two demensions")
        return None

    # blockSize检查
    if blockSize % 2 == 0 or blockSize < 3:
        print('blockSize is not odd or too small')
        return None
    # print('blockSize', blockSize)
    # 计算addSize
    addSize = int((blockSize - 1) / 2)
    newHeight = img.shape[0] + blockSize - 1
    newWidth = img.shape[1] + blockSize - 1

    # 中间结果
    imgMiddle = np.zeros((newHeight, newWidth))
    imgMiddle[:, :] = 255
    # print('imgMiddle',imgMiddle)
    # print('type(newHeight)',type(newHeight))
    # print('type(addSize)',type(addSize))
    imgMiddle[addSize:newHeight - addSize, addSize:newWidth - addSize] = img
    # print('imgMiddle', imgMiddle)
    imgDark = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    localMin = 255

    for i in range(addSize, newHeight - addSize):
        for j in range(addSize, newWidth - addSize):
            localMin = 255
            for k in range(i - addSize, i + addSize + 1):
                for l in range(j - addSize, j + addSize + 1):
                    if imgMiddle.item((k, l)) < localMin:
                        localMin = imgMiddle.item((k, l))
            imgDark[i - addSize, j - addSize] = localMin

    return imgDark


# 获取最小值矩阵
# 获取BGR三个通道的最小值
def getMinChannel(img):
    # 输入检查
    if len(img.shape) == 3 and img.shape[2] == 3:
        pass
    else:
        print("bad image shape, input must be color image")
        return None
    imgGray = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    localMin = 255

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            localMin = 255
            for k in range(0, 3):
                if img.item((i, j, k)) < localMin:
                    localMin = img.item((i, j, k))
            imgGray[i, j] = localMin
    return imgGray



θ = 0.2 #θ is a parameter controlling the selection of the bright parts in the red channel.
σ = 0.3  #σ is a parameter of a two-dimensional Gaussian function 
k = 0.5  #k值暂定
N = 15 #N*N的大小

img = cv2.imread(r'C:\Users\Owen\Pictures\HCF-test.jpg')
#cv_show('original',img)



original = img.copy()
original_img = original.copy()

B,G,R =cv2.split(original)#拆分图像通道
L = (B+G+R) / 3  #求均值
#cv_show('R',original[:,:,0])

RGB_merge('original',B,G,R)

original = np.float64(original)
L1_normalized = cv2.normalize(original,dst = None,alpha = 0,beta = 1 , norm_type = cv2.NORM_MINMAX)  #归一化

HCFB,HCFG,HCFR = mat_mean(L1_normalized,θ,N) #各通道均值

HCFL = L_mean(L,N)  #L均值

HCF = cv2.merge([HCFB,HCFG,HCFR])

HCF = HCF * 255
HCF = np.uint8(HCF)
HCF = cv2.GaussianBlur(HCF,(N,N),σ)
cv_show('HCF',HCF)


print(HCFB[int(HCFB.shape[0]/2),int(HCFB.shape[1]/2)])


#with the HC feedbacks,the cone signals become:
CSR = R / zero_to_one(HCF[:,:,0])  
CSG = G / zero_to_one(HCF[:,:,1])
CSB = B / zero_to_one(HCF[:,:,2])
CSL = L / HCFL

RGB_merge('CS IMG',CSR,CSG,CSB)

#use the sigmoid function to suppress the dim part of cone signal:
CSR = CSR - 0.5
CSG = CSG - 0.5
CSB = CSB - 0.5
CSL = CSL - 0.5


COR=1/(1+np.exp(-10*CSR))#the output of rod cells COrod（x,y）
COG=1/(1+np.exp(-10*CSG))
COB=1/(1+np.exp(-10*CSB))
COrod = 1/(1+np.exp(-10*CSL))

RGB_merge('CO IMG',COR,COG,COB)

#fbcL and fbsL are respectively the signals of rod cells and the signals
#of rod cells after modulation by the rod horizontal cells
FbcL = COrod 
FbsL = L_mean(COrod,N)

#the output of rod bipolar cells:
Bprod = max_mat(0,np.float32(cv2.GaussianBlur(FbcL,(5,5),sigmaX=σ) - k * cv2.GaussianBlur(FbsL,(5,5),sigmaX=3*σ)))


Bprod_gamma = exposure.adjust_gamma(Bprod,gamma=0.5) #use the gamma correction to simulate the nonlinear  modulation of amacrine cells.

#the input to the RF center of cone bipolar cells fbc is given by:
FbcR = COR * Bprod_gamma  #(10)
FbcG = COG * Bprod_gamma
FbcB = COB * Bprod_gamma

#fbs is the cone signal locally processed by the horizontal cells, which
#is treated as the input to the RF surround of cone bipolar cells

FbsR = L_mean(COR,N)   #（11）
FbsG = L_mean(COG,N)
FbsB = L_mean(COB,N)


#the signals along the OFF pathway fbc and fbs can  be simply obtained by:
#(12)

ones_mat = np.ones(shape = FbcR.shape,dtype=np.uint8)

FbcR_1 = ones_mat - FbcR #(12)
FbcG_1 = ones_mat - FbcG
FbcB_1 = ones_mat - FbcB

#(7)
BpR = max_mat(0,np.float32(cv2.GaussianBlur(FbcR,(5,5),sigmaX=σ) - k * cv2.GaussianBlur(FbsR,(5,5),sigmaX = 3 * σ)))
BpB = max_mat(0,np.float32(cv2.GaussianBlur(FbcB,(5,5),sigmaX=σ) - k * cv2.GaussianBlur(FbsB,(5,5),sigmaX = 3 * σ)))
BpG = max_mat(0,np.float32(cv2.GaussianBlur(FbcG,(5,5),sigmaX=σ) - k * cv2.GaussianBlur(FbsG,(5,5),sigmaX = 3 * σ)))

BpR_ = max_mat(0,np.float32(cv2.GaussianBlur(FbcR_1,(5,5),sigmaX=σ) - k*cv2.GaussianBlur(FbsR_1,(5*5),sigmaX=3 * σ))) 
BpB_ = max_mat(0,np.float32(cv2.GaussianBlur(FbcR_1,(5,5),sigmaX=σ) - k*cv2.GaussianBlur(FbsR_1,(5*5),sigmaX=3 * σ)))
BPG_ = max_mat(0,np.float32(cv2.GaussianBlur(FbcG,(5,5),sigmaX=σ) - k * cv2.GaussianBlur(FbsG,(5,5),sigmaX = 3 * σ)))
#(13)

Bpy = (BpR + BpB) / 2
Bpy_ = (BpR_ + BpB_) / 2



