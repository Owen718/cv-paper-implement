import cv2
import numpy as np
import GuidedFilter


def cv_show(img_name,img):  #显示图像以便观察，按任意键继续
    cv2.imshow(img_name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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


img = cv2.imread(r'C:\Users\Owen\Pictures\Haze_Removal_test10.jpg')

N=15  #窗口大小，窗口大小=最小滤波半径*2 + 1
Radius = 5
W=0.95 #透射率t的修正系数
t0=0.1  #透射率t的阈值



imgGray=getMinChannel(img)
dark_channel = getDarkChannel(imgGray,N)

#dark_channel=min_functin(Radius,Radius,dark_channel)
#dark_channel=cv2.GaussianBlur(dark_channel,ksize=(Radius,Radius),sigmaX=0.3)
cv_show('dark_channel',dark_channel)

#求全球大气光A值：
#1.遍历暗通道，寻找大小为前0.1%的像素：
#max_dark_index=np.argmax(dark_channel)
#max_dark_x=int(max_dark_index/dark_channel.shape[1])
#max_dark_y=int(max_dark_index%dark_channel.shape[1])
#print(max_dark_x,max_dark_y)

dark_sorted = [i for j in dark_channel for i in j]  #二维快速降至一维
dark_sorted = sorted(dark_sorted,reverse=True)
piexl_toal = len(dark_sorted)
max_dark_num = dark_sorted[0:int(piexl_toal/1000)]  #取前0.1%
max_dark_num = list(set(max_dark_num))  #使用set去重
#print(max_dark_num)

#2.遍历前0.1%的像素在原图中的大小，取最大值,作为全球大气光值A。
atmosphericLightB=0
atmosphericLightG=0
atmosphericLightR=0
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)



for y in range(dark_channel.shape[0]):
    for x in range(dark_channel.shape[1]):
        for dark_num in max_dark_num:
            if dark_channel[y,x]==dark_num and img[y,x,0]>atmosphericLightB:     
                atmosphericLightB=img[y,x,0]
            if dark_channel[y,x]==dark_num and img[y,x,1]>atmosphericLightG:
                atmosphericLightG=img[y,x,1]
            if dark_channel[y,x]==dark_num and img[y,x,2]>atmosphericLightR:
                atmosphericLightR=img[y,x,2]

if atmosphericLightB > 220:
    atmosphericLightB = 220
if atmosphericLightG > 220:
    atmosphericLightG = 220
if atmosphericLightR > 220:
    atmosphericLightR = 220



#atmosphericLight = max(atmosphericLightR,atmosphericLightG,atmosphericLightB)

dark_channel = np.float64(dark_channel)



#用暗通道来求透射率t(x)的预估值：
dark_channel = np.float64(dark_channel)
transmission = 1 - W * dark_channel / max(atmosphericLightB,atmosphericLightG,atmosphericLightR)

gimfiltR = 50
eps = 10 ** -3

#导向滤波测试
guided_filter = GuidedFilter.GuidedFilter(img,gimfiltR,eps)
transmission = guided_filter.filter(transmission)
#防止出现t小于0的情况
transmission = np.clip(transmission, t0, 0.9)

cv_show('transmission',transmission)



img=np.float64(img)
J_hazed_removal_B = (img[:,:,0] - atmosphericLightB) / transmission  +  atmosphericLightB
J_hazed_removal_G = (img[:,:,1] - atmosphericLightG) / transmission  +  atmosphericLightG
J_hazed_removal_R = (img[:,:,2] - atmosphericLightR) / transmission  +  atmosphericLightR
J_hazed_removal = cv2.merge([J_hazed_removal_B,J_hazed_removal_G,J_hazed_removal_R])

J_hazed_removal = np.clip(J_hazed_removal,0,255)
J_hazed_removal = np.uint8(J_hazed_removal)


cv_show('original',img)
cv_show('J_hazed_removal',J_hazed_removal)
