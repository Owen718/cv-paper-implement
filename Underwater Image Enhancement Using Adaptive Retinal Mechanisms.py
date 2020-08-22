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
            if xy_zone.max() < θ:  #若是最大值小于θ
                R_mean[x,y]=xy_zone.max()
                max_num += 1
                xy_zone=[]
            else:  #若是最大值小于θ
                mean_num += 1
                R_mean[x,y]=np.mean(xy_zone[xy_zone>θ])
                #original_view = cv2.circle(original_view,center=(y,x),radius = 1,color = (255,255,255))#绘制区域
                original_view[x,y,:] = [255,0,0]
                xy_zone=[]
     
                
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
def getMinChannel(img):  #return np.float64 imgGray
    # 输入检查
    if len(img.shape) == 3 and img.shape[2] == 3:
        pass
    else:
        print("bad image shape, input must be color image")
        return None
    imgGray = np.zeros((img.shape[0], img.shape[1]), dtype=np.float64)
    localMin = 999999

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            localMin = 999999
            for k in range(0, 3):
                if img.item((i, j, k)) < localMin:
                    localMin = img.item((i, j, k))
            imgGray[i, j] = localMin
    return imgGray



θ = 0.3 #θ is a parameter controlling the selection of the bright parts in the red channel.
σ = 0.3  #σ is a parameter of a two-dimensional Gaussian function 
k = 0.5  #k值暂定
N = 49 #N*N的大小

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

HCF_255 = HCF * 255
HCF_255= np.uint8(HCF_255)
HCF_255= cv2.GaussianBlur(HCF_255,(N,N),σ)
cv_show('HCF',HCF_255)


#with the HC feedbacks,the cone signals become:
CSR = R / HCF[:,:,0]    #存在问题！ HCF含有0  #(4)
CSG = G / HCF[:,:,1]
CSB = B / HCF[:,:,2]
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

middel = np.float64(getMinChannel(img)) / np.float64(getMinChannel(HCF))  #(15)
k = np.mean(middel)   #(17) k要先算


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

ones_mat = np.ones(shape = FbcR.shape,dtype=np.float64)

FbcR_1 = np.zeros(shape = FbcR.shape,dtype=np.float64)
FbcG_1 = np.zeros(shape = FbcR.shape,dtype=np.float64)
FbcB_1 = np.zeros(shape = FbcR.shape,dtype=np.float64)
FbsR_1 = np.zeros(shape = FbcR.shape,dtype=np.float64)
FbsG_1 = np.zeros(shape = FbcR.shape,dtype=np.float64)
FbsB_1 = np.zeros(shape = FbcR.shape,dtype=np.float64)

FbcR_1 = ones_mat - np.float64(FbcR) #(12)
FbcG_1 = ones_mat - np.float64(FbcG)
FbcB_1 = ones_mat - np.float64(FbcB)

FbsR_1 = ones_mat - np.float64(FbsR)
FbsG_1 = ones_mat - np.float64(FbsG)
FbsB_1 = ones_mat - np.float64(FbsB)


#(7)
BpR = max_mat(0,np.float64(cv2.GaussianBlur(FbcR,(5,5),sigmaX=σ) - k * cv2.GaussianBlur(FbsR,(5,5),sigmaX = 3 * σ)))
BpB = max_mat(0,np.float64(cv2.GaussianBlur(FbcB,(5,5),sigmaX=σ) - k * cv2.GaussianBlur(FbsB,(5,5),sigmaX = 3 * σ)))
BpG = max_mat(0,np.float64(cv2.GaussianBlur(FbcG,(5,5),sigmaX=σ) - k * cv2.GaussianBlur(FbsG,(5,5),sigmaX = 3 * σ)))

BpR_ = max_mat(0,np.float64(cv2.GaussianBlur(FbcR_1,(5,5),sigmaX=σ) - k*cv2.GaussianBlur(FbsR_1,(5,5),sigmaX = 3 * σ))) 
BpB_ = max_mat(0,np.float64(cv2.GaussianBlur(FbcR_1,(5,5),sigmaX=σ) - k*cv2.GaussianBlur(FbsR_1,(5,5),sigmaX = 3 * σ)))
BPG_ = max_mat(0,np.float64(cv2.GaussianBlur(FbcG_1,(5,5),sigmaX=σ) - k*cv2.GaussianBlur(FbsG_1,(5,5),sigmaX = 3 * σ)))

#(13)
Bpy = (BpR + BpB) / 2
Bpy_ = (BpR_ + BpB_) / 2


#(15)

t = np.float64(ones_mat) - getDarkChannel(middel,7)   #求t值 
cv_show('t',t)



Bp = cv2.merge([BpR,BpG,BpB])
S = 3 * getMinChannel(Bp)/ (BpR+BpG+BpB)  #？？ 存在问题，BpR/G/B中含有0
 #？？ 存在问题，BpR/G/B中含有0  #HSV色彩空间，S值
                                                                        #逻辑值索引，在不是0的地方才改变其的值

#m的值与双极单元输出的饱和程度成反比。饱和度越低，直观地表明m值越高，因此周围RF对中心RF的贡献越大，有助于增强水下图像的色彩对比度。
m = np.mean(S) 
S = ones_mat - S  #图像的饱和度可以作为图像颜色对比度的简单测量

fgc = BpR
fgs = BpG

fgc_ = BpR_
fgs_ = BPG_

Gg = max_mat(0,cv2.GaussianBlur(fgc,(N,N),sigmaX=σ)+ m * (cv2.GaussianBlur(fgc,(N,N),sigmaX=σ)-cv2.GaussianBlur(fgs,(N,N),sigmaX= 3 * σ)))
Gg_ = max_mat(0,cv2.GaussianBlur(fgc_,(N,N),sigmaX=σ)+ m * (cv2.GaussianBlur(fgc_,(N,N),sigmaX=σ)-cv2.GaussianBlur(fgs_,(N,N),sigmaX=3 * σ)))

 
F_G = 1 / (1 + np.exp(-10*(Gg-0.5)))
Won = F_G / (Gg + F_G)
Woff = ones_mat - Won

output = Won * Gg + Woff * Gg_



