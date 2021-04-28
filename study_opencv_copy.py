import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
import os
# from locating import locating
# 回调函数，x表示滑块的位置，本例暂不使用
def nothing(x):
    pass

width_min=238
height_min=70
area_min=2000#对于不同分辨率的车牌图像这个值是不定的

def locating(rawImage):
    # 读取图片
    # rawImage = cv2.imread(path)
    # rawImage = cv2.resize(rawImage, dsize=(1280, 960))

    #### 颜色、饱和度滤波 ####
    # 掩膜：BGR通道，若像素B分量在 100~255 且 G分量在 0~190 且 G分量在 0~140 置255（白色） ，否则置0（黑色）
    mask_gbr = cv2.inRange(rawImage, (100, 0, 0), (255, 190, 140))  # (100, 0, 0), (255, 190, 140)

    img_hsv = cv2.cvtColor(rawImage, cv2.COLOR_BGR2HSV)  # 转换成 HSV 颜色空间
    h, s, v = cv2.split(img_hsv)  # 分离通道  色调(H)，饱和度(S)，明度(V)
    mask_s = cv2.inRange(s, 130, 255)  # 取饱和度通道进行掩膜得到二值图像  (,95,255)

    rgbs = mask_gbr & mask_s  # 与操作，两个二值图像都为白色才保留，否则置黑
    # 核的横向分量大，使车牌数字尽量连在一起
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 3))  # (15, 5)
    img_rgbs_dilate = cv2.dilate(rgbs, kernel, 3)  # 膨胀 ，减小车牌空洞

    img_blue = cv2.medianBlur(img_rgbs_dilate, 15)
    # cv2.imshow('filter', img_blue)

    #### 边缘识别 ####
    # 高斯模糊，将图片平滑化，去掉干扰的噪声
    img = cv2.GaussianBlur(rawImage, (3, 3), 0)  # （3，3）
    # 图片灰度化
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Sobel算子（X方向）
    Sobel_x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    absX = cv2.convertScaleAbs(Sobel_x)  # 转回uint8
    img = absX
    # 二值化：图像的二值化，就是将图像上的像素点的灰度值设置为0或255,图像呈现出明显的只有黑和白
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    # 闭操作：闭操作可以将目标区域连成一个整体，便于后续轮廓的提取。
    kernelX = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (160, 30))    # （15，3）
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernelX)
    # cv2.imshow('close', image)
    # 膨胀腐蚀(形态学处理)
    kernelX = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 15))    # （15，3）
    kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 15))    # （10，3）
    img = cv2.dilate(img, kernelX, iterations=2)
    img = cv2.erode(img, kernelX, iterations=2)     # 1
    # image = cv2.dilate(img, kernelX)
    # img = cv2.erode(img, kernelX, iterations=1)     # 1
    # image = cv2.erode(image, kernelY)
    # img = cv2.dilate(img, kernelX)
    img_edge = cv2.dilate(img, kernelY, iterations=3)   # 3
    # cv2.imshow('edge', img_edge)
    # 平滑处理，中值滤波
    img_edge = cv2.medianBlur(img, 15)

    image = img_blue & img_edge
    # image = cv2.dilate(image, kernelY, iterations=1)     # 1
    image = cv2.medianBlur(image, 25)   # 15
    # cv2.imshow('conbine', image)

    # 查找轮廓
    contours, w1 = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    all_chepai = []
    all_area = []
    max_area = 0
    real_chepai = None

    for item in contours:
        rect = cv2.boundingRect(item)
        contour_area = cv2.contourArea(item)
        min_rect = cv2.minAreaRect(item)
        # area = min_rect[1][0] * min_rect[1][1]

        x = rect[0]
        y = rect[1]
        weight = rect[2]
        height = rect[3]
        area = weight * height

        if weight > 1.4 * height and 12800 < area < 400000:    # 800 < area < 25000
            # 裁剪区域图片
            chepai = rawImage[y - 20:y + height + 20, x - 20:x + weight + 20]
            if contour_area > max_area:
                real_chepai = chepai
                max_area = contour_area
            all_chepai.append(chepai)
            # all_area.append(area)
            # cv2.imshow('chepai', chepai)

    # real_chepai = all_chepai[all_area.index(max(all_area))]

    # 绘制轮廓
    # image = cv2.drawContours(rawImage, contours, -1, (0, 0, 255), 3)
    # image = cv2.resize(image,(1280, 960))
    # cv2.imshow('image', image)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    if (real_chepai is None)|(np.size(real_chepai)==0):
        real_chepai=rawImage


    return real_chepai


def getPerspectiveTransform(raw_img):
    #### 颜色、饱和度滤波 ####
    # 掩膜：BGR通道，若像素B分量在 100~255 且 G分量在 0~190 且 G分量在 0~140 置255（白色） ，否则置0（黑色）
    mask_gbr = cv2.inRange(raw_img, (100, 0, 0), (255, 190, 140))  # (100, 0, 0), (255, 190, 140)

    img_hsv = cv2.cvtColor(raw_img, cv2.COLOR_BGR2HSV)  # 转换成 HSV 颜色空间
    h, s, v = cv2.split(img_hsv)  # 分离通道  色调(H)，饱和度(S)，明度(V)
    mask_s = cv2.inRange(s, 130, 255)  # 取饱和度通道进行掩膜得到二值图像  (,95,255)

    rgbs = mask_gbr & mask_s  # 与操作，两个二值图像都为白色才保留，否则置黑
    # 核的横向分量大，使车牌数字尽量连在一起
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))  # (15, 5)
    img_rgbs_dilate = cv2.dilate(rgbs, kernel, 3)  # 膨胀 ，减小车牌空洞

    img_blue = cv2.medianBlur(img_rgbs_dilate, 15)
    # cv2.imshow('filter', img_blue)
    img = img_blue

    #### 边缘识别 ####
    # 高斯模糊，将图片平滑化，去掉干扰的噪声
    img = cv2.GaussianBlur(raw_img, (3, 3), 0)  # （3，3）
    # 图片灰度化
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Sobel算子（X方向）
    Sobel_x = cv2.Sobel(img, cv2.CV_16S, 2, 0)
    absX = cv2.convertScaleAbs(Sobel_x)  # 转回uint8
    img = absX
    # 二值化：图像的二值化，就是将图像上的像素点的灰度值设置为0或255,图像呈现出明显的只有黑和白
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    # cv2.imshow('sober',img)
    # 闭操作：闭操作可以将目标区域连成一个整体，便于后续轮廓的提取。
    kernelX = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 3))    # （15，3）
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernelX)
    # cv2.imshow('close', img)
    # 膨胀腐蚀(形态学处理)
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))    # （15，3）
    kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))    # （10，3）
    img = cv2.dilate(img, kernelX, iterations=3)
    img = cv2.erode(img, kernelX, iterations=2)     # 1
    # image = cv2.dilate(img, kernelX)
    # img = cv2.erode(img, kernelX, iterations=1)     # 1
    # image = cv2.erode(image, kernelY)
    # img = cv2.dilate(img, kernelX)
    # img = cv2.dilate(img, kernelY, iterations=1)   # 3
    # cv2.imshow('edge', img)
    img = img & img_blue
    # 平滑处理，中值滤波
    img = cv2.medianBlur(img, 15)



    # 查找轮廓
    contours, w1 = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0

    for item in contours:
        rect = cv2.minAreaRect(item)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        length = raw_img.shape[1]
        width = np.int0(length/np.pi)
        angle = abs(rect[2])
        area = cv2.contourArea(item)

        if area > 800 and area >= max_area:
            max_area = area
            # 获取四个顶点坐标

            left_point_x = np.min(box[:, 0])
            right_point_x = np.max(box[:, 0])
            top_point_y = np.min(box[:, 1])
            bottom_point_y = np.max(box[:, 1])

            left_point_y = box[:, 1][np.where(box[:, 0] == left_point_x)][0]
            right_point_y = box[:, 1][np.where(box[:, 0] == right_point_x)][0]
            top_point_x = box[:, 0][np.where(box[:, 1] == top_point_y)][0]
            bottom_point_x = box[:, 0][np.where(box[:, 1] == bottom_point_y)][0]
            # 上下左右四个点坐标
            vertices = np.array([[top_point_x, top_point_y], [bottom_point_x, bottom_point_y],
                                 [left_point_x, left_point_y], [right_point_x, right_point_y]], dtype=np.float32)

            left_length = np.sqrt((top_point_x - left_point_x) ** 2 + (top_point_y - left_point_y) ** 2)
            right_length = np.sqrt((top_point_x - right_point_x) ** 2 + (top_point_y - right_point_y) ** 2)

            if angle < 8:
                result = raw_img
            else:
                if left_length > right_length:
                        desired = np.array([[length, 0], [0, width], [0, 0], [length, width]], dtype=np.float32)
                        M = cv2.getPerspectiveTransform(vertices, desired)
                        result = cv2.warpPerspective(raw_img, M, (length, width))
                else:
                        desired = np.array([[0, 0], [length, width], [0, width], [length, 0]], dtype=np.float32)
                        M = cv2.getPerspectiveTransform(vertices, desired)
                        result = cv2.warpPerspective(raw_img, M, (length, width))
            # cv2.imshow('result', result)

    # 绘制轮廓
    # image = cv2.drawContours(raw_img, contours, -1, (0, 0, 255), 3)
    # image = cv2.resize(image, (1280, 960))
    # cv2.imshow('image', image)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    if (result is None)|(np.size(result)==0):
        result=raw_img

    return result


# path="./data/sub_test_pics/"


def prepros(path):
    exp=cv2.imdecode(np.fromfile(os.path.join("./","exp.jpg"),dtype=np.uint8),1)
    exph,expw= exp.shape[:2]
    dirlist=os.listdir(path)#这里的代码用于批量测试
    for imgname in dirlist:
        if(imgname[-4:]=='.jpg')|(imgname[-4:]=='.JPG'):
            img=cv2.imdecode(np.fromfile(os.path.join(path,imgname),dtype=np.uint8),1)
            # img=cv2.imread(path+imgname)
            print(imgname)
            img=locating(img)
            cv2.imwrite("./data/check/"+imgname,img)
            img=getPerspectiveTransform(img)
        #     img=cv2.resize(img,(expw,exph),interpolation=cv2.INTER_CUBIC)
        #     # cv2.imwrite("../data/res/"+imgname[0:-4]+".jpg", img)
        #     # cv2.imwrite("./data/pre_res/"+imgname, img)
        #     height=img.shape[0]
        #     width=img.shape[1]
        #     # if(height<height_min and width<width_min):
        #     #     img = cv2.resize(img, (width_min, height_min), interpolation=cv2.INTER_LANCZOS4)
        #     #     height=height_min
        #     #     width=width_min
        #     img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        #     area_min=height*width*0.002
        #     contours,boxes = search_in_hsv(img_hsv)
        #     rois=roi(boxes,img)
        
        # #这里的代码用于批量测试
        #     cnt=0
        #     for img_cut in rois:
        #         if img_cut.size>0:
        #             cnt=cnt+1
        #             img_cut=cv2.resize(img_cut,(expw,exph))
        #             cv2.imwrite("../data/res/"+imgname[0:-4]+"-"+str(cnt)+".jpg", img_cut)
            img=cv2.resize(img,(expw,exph))
            if not os.path.exists("./data/res/"):
                os.makedirs("./data/res/")
            cv2.imwrite("./data/res/"+imgname, img)    

            



            
# python study_opencv.py
        


    