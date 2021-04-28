import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
import os
from skimage import morphology

width_min=512
height_min=384
area_min=2000#对于不同分辨率的车牌图像这个值是不定的

#数据格式解释：图像的x轴是横向的（宽度），y轴是纵向的（高度），像素点坐标格式为[x y]
#roi功能：代入的坐标格式是img[y_min:y_max,x_min:x_max]，注意第一项是y轴坐标
#shape[0]返回高度，shape[1]返回宽度

def print_img_in_allcolorspace(img):
    img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow('raw',img)
    cv2.imshow('gray',img_gray)
    cv2.imshow('b',img[:,:,0])
    cv2.imshow('g',img[:,:,1])
    cv2.imshow('r',img[:,:,2])
    img_h=img_hsv[:,:,0]
    cv2.imshow('h',img_h)
    img_s=img_hsv[:,:,1]
    cv2.imshow('s',img_s)
    img_v=img_hsv[:,:,2]
    cv2.imshow('v',img_v)


def search_in_hsv(img):
    img_rgb=cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))  # 椭圆结构
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)) 
    kernel_large = np.ones((8, 8), np.uint8)

    img_hsv_s=img[:,:,1]#取出S平面
    # img_hsv_s=cv2.GaussianBlur(img_hsv_s, (5, 5), 1)
    img_hsv_s = cv2.morphologyEx(img_hsv_s, cv2.MORPH_CLOSE, kernel)
    hist = np.bincount(img_hsv_s.ravel(), minlength=256)
    histSum = np.sum(hist)
    print('直方图总和：',histSum)
    thVal=0
    valSum=0
    for i in range(255,0,-1):
        valSum=valSum+hist[i]
        if(valSum>histSum*0.02):
            thVal=i
            print('二值化阈值为：',thVal)
            break
    ret,th=cv2.threshold(img_hsv_s,thVal,255,cv2.THRESH_BINARY)#二值化
    # plt.figure(0)
    # plt.imshow(th)
    # plt.show()
    erosion = cv2.erode(th, kernel_small)  # 腐蚀
    dilation = cv2.dilate(erosion, kernel)#膨胀

    try:#findContours寻找白色轮廓（必须保证背景是黑色）
        contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#contours为双重向量组，每个向量元素保存一组可能是边缘的点集
    except ValueError:																			 #hierarchy与contours相对应，hierarchy[i][0] ~hierarchy[i][3]，分别表示第i个轮廓的后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号
        image, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_rgb, contours, -1, (255, 0, 0), 2)#调试：预览绘制的轮廓
    contours_find=[]
    box_find=[]
    rois=[]
    for cnt in contours:
        if cv2.contourArea(cnt) > area_min:#contourArea用于计算轮廓的面积。计算每一个轮廓区域的面积并去掉过小的区域。
            rect = cv2.minAreaRect(cnt)#返回可以包围输入轮廓的最小外接矩形（矩形可以斜着），返回值为Box2D结构：（0最小外接矩形的中心（x，y），1（宽度，高度），2度数制的旋转角度）
            print('rect:',rect)
            contours_find.append(rect)
            box = cv2.boxPoints(rect)#根据矩形的角度不同，返回的顶点的顺序会有不同
            box = np.int0(box)
            tmp = roi2(rect,box,img)
            if(np.any(tmp)):
                rois.append(tmp)


    # plt.figure(1)
    # plt.plot(hist)
    # plt.figure(2)
    # plt.imshow(img_rgb)
    # plt.show()
    print('【S域搜索】在S域下找到了%d个可能的轮廓'%len(rois))
    return contours_find,rois

def rotate(ps,m):
    pts = np.float32(ps).reshape([-1, 2])  # 要映射的点
    pts = np.hstack([pts, np.ones([len(pts), 1])]).T
    target_point = np.dot(m, pts)
    target_point = np.int0([[target_point[0][x],target_point[1][x]] for x in range(len(target_point[0]))])
    return target_point

def roi2(rect,box,img):
    cols=img.shape[1]
    rows=img.shape[0]
    angle=-(90-rect[2])
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    if (rect[2]==0):
        box_rotated=box
    else:
        box_rotated=rotate(box,M)#获得旋转之后的矩形点
    # print(box)
    # print(box_rotated)
    # cv2.drawContours(dst, [box_rotated], -1, (0, 255, 0), 2)#调试：预览绘制的最小外接矩形
    roi_return=dst[box_rotated[0][1]:box_rotated[2][1],box_rotated[0][0]:box_rotated[2][0]]
    if(np.any(roi_return)):
        if(roi_return.shape[0]>roi_return.shape[1]):
            roi_return=cv2.flip(cv2.transpose(roi_return),0)#图像顺时针旋转270度
    return roi_return

def roi(box,img):#从图片中截取给定的区域，返回包含给定区域图片的列表
    ans=[]
    for rect in box:
        x1=0
        y1=0
        x2=img.shape[0]
        y2=img.shape[1]
        dot_2=[]
        # print(rect)
        # print(x1,y1,x2,y2)
        dot_2=copy.deepcopy(rect)
        dot_2[1][1]=rect[0][1]#把矩形的右上角强制拉到和左上角相同高度的位置
        # dot_2[3][0]=rect[0][0]#把矩形的左下角强制拉到和左上角相同宽度的位置
        x1=dot_2[0][0]
        y1=dot_2[0][1]
        x2=dot_2[1][0]
        y2=dot_2[3][1]
        # print(dot_2)
        raw = np.float32([rect[0], rect[1], rect[3]])
        trans = np.float32([dot_2[0], dot_2[1], dot_2[3]])
        # print(raw)
        # print(trans)
        M = cv2.getAffineTransform(raw, trans)#计算仿射变换矩阵
        aftertrans = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))#对原图像进行仿射变换，返回变换后的图像，输出的尺寸仍为原图像尺寸
        # cv2.imshow('aftertrans',aftertrans)
        img_cut=aftertrans[y1:y2,x1:x2]
        if(img_cut.shape[0]<=img_cut.shape[1]):#如果变形结束图片还是立着的，就排除
            ans.append(img_cut)
        # ans.append(img[y2:y1,x2:x1])
    # for i in ans:
    #     cv2.imshow('ans',i)
    #     cv2.waitKey(0)
    print('【照片截取】从原图中截取了%d个有效对象'%len(ans))
    return ans

def color_recog(imgs):#输入hsv平面的图片集
    clr=[]
    cnt=-1
    dellist=[]
    for img in imgs:
        cnt=cnt+1
        img_h=img[:,:,0]
        hist = np.bincount(img_h.ravel(), minlength=256)
        # print(hist)
        # plt.plot(hist)
        # plt.show()
        pos=np.argwhere(hist==np.max(hist))
        print('maxVal Color:',pos)
        pos=pos[0]
        if(pos>93 and pos<135):
            clr.append('B')
            continue
        if(pos>14 and pos<32):
            clr.append('Y')
            continue
        if(pos>32 and pos<82):
            clr.append('G')
            continue
        dellist.append(cnt)#识别不出颜色的记下来，后面去掉
        # clr.append('X')
    print('【颜色识别】颜色识别结果：',clr)
    cnt=0
    for i in dellist:
        imgs.pop(i-cnt)
        cnt=cnt+1
    print('【颜色识别】识别了%d个图像的颜色'%len(clr))
    return clr
    
def contourProcess(imgs):#输入HSV平面的图像集
    edges=[]
    for img in imgs:
        # cv2.imshow('edges_nowImage',img)
        img_v=img[:,:,2]
        # img_v= cv2.resize(img_v,(300,100))
        img_v=cv2.bilateralFilter(img_v, 5, 75, 75)
        # cv2.imshow('GaussianBlur',img_v)
        edge = cv2.Canny(img_v, 40, 90)
        
        img_v= cv2.resize(img_v,(440,140))#统一输出图片尺寸
        cv2.imshow('img_v_edges',edge)
        edges.append(edge)
        cv2.waitKey(0)

    return edges

def accurate_locate(imgs):
    edges=[]
    for img in imgs:
        img_s=img[:,:,1]
        img_s=cv2.bilateralFilter(img_s, 5, 75, 75)
        img_rgb=cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
        ret, th = cv2.threshold(img_s, 100, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 椭圆结构

        #获得文字边缘
        img_contours = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
        contours, hierarchy = cv2.findContours(th,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_contours, contours, -1, 255, 1)

        edge = cv2.Canny(th, 45, 90)
        lines = cv2.HoughLinesP(img_contours, 1, np.pi / 180, 1, minLineLength=10, maxLineGap=1)
        slopes=[]
        if(lines is None):#如果没找到线就跳过
            continue
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if(x2!=x1):
                slope=(y2-y1)/(x2-x1)
            else:
                slope=1000#垂直情况
            if(abs(slope)>=0.1):
                slopes.append(slope)
                print(slope)
            cv2.line(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 1, lineType=cv2.LINE_AA)        
        if not(slopes):#如果找到的线的斜率都过小则跳过
            continue
        slope_mean=np.mean(slopes)
        print('【透视校正】平均斜率：',slope_mean)
        M = np.float32([[1, -1/slope_mean, 0],#x轴的剪切：图像的上面不动，正值代表下方向右变形，负值代表向左变形。数值越大变形程度越大。
                        [0, 1  , 0],
                        [0, 0  , 1]])
        edge_sheared = cv2.warpPerspective(img_contours,M,(int(img.shape[1]),int(img.shape[0])))
        img_sheared = cv2.warpPerspective(img_rgb,M,(int(img.shape[1]),int(img.shape[0])))
        x, y, w, h = cv2.boundingRect(edge_sheared)  # 外接矩形
        img_sheared = img_sheared[y:y+h,x:x+w] #截取图片中的有效区域
        img_sheared = cv2.cvtColor(img_sheared,cv2.COLOR_RGB2BGR)
        edges.append(img_sheared)
        print(M)
        # plt.figure(1)
        # plt.imshow(img_sheared)
        # plt.show()
    return edges




def locating(rawImage,block,desize):
    # 读取图片
    # rawImage = cv2.imread(path)
    if desize==True:
        Image = cv2.resize(rawImage, dsize=(1000, 800))

    # 掩膜：BGR通道，若像素B分量在 100~255 且 G分量在 0~190 且 G分量在 0~140 置255（白色） ，否则置0（黑色）
    mask_gbr = cv2.inRange(rawImage, (100, 0, 0), (255, 190, 140))

    img_hsv = cv2.cvtColor(rawImage, cv2.COLOR_BGR2HSV)  # 转换成 HSV 颜色空间
    h, s, v = cv2.split(img_hsv)  # 分离通道  色调(H)，饱和度(S)，明度(V)
    mask_s = cv2.inRange(s, 130, 255)  # 取饱和度通道进行掩膜得到二值图像

    rgbs = mask_gbr & mask_s  # 与操作，两个二值图像都为白色才保留，否则置黑
    # 核的横向分量大，使车牌数字尽量连在一起
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 5))
    img_rgbs_dilate = cv2.dilate(rgbs, kernel, 3)  # 膨胀 ，减小车牌空洞

    img_blue = cv2.medianBlur(img_rgbs_dilate, 15)
    # cv2.imshow('filter', img_blue)

    # 高斯模糊，将图片平滑化，去掉干扰的噪声
    img = cv2.GaussianBlur(rawImage, (3, 3), 0)
    # 图片灰度化
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Sobel算子（X方向）
    Sobel_x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    absX = cv2.convertScaleAbs(Sobel_x)  # 转回uint8
    img = absX
    # 二值化：图像的二值化，就是将图像上的像素点的灰度值设置为0或255,图像呈现出明显的只有黑和白
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    # 闭操作：闭操作可以将目标区域连成一个整体，便于后续轮廓的提取。
    kernelX = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernelX)
    # cv2.imshow('close', image)
    # 膨胀腐蚀(形态学处理)
    kernelX = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 3))
    kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    # img = cv2.dilate(img, kernelX)
    img = cv2.erode(img, kernelX, iterations=1)
    # image = cv2.dilate(image, kernelX)
    # img = cv2.erode(img, kernelX, iterations=1)
    # image = cv2.erode(image, kernelY)
    img = cv2.dilate(img, kernelX, iterations=2)
    img_edge = cv2.dilate(img, kernelY, iterations=3)
    # cv2.imshow('fushi', img_edge)
    # 平滑处理，中值滤波
    # img_edge = cv2.medianBlur(img, 15)

    image = img_blue & img_edge
    image = cv2.medianBlur(image, 15)
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

        if weight > 1.2 * height  : #and 200 < area< 27000
            # 裁剪区域图片
            chepai = rawImage[y - block:y + height + block, x -block:x + weight + block]
            if contour_area > max_area:
                real_chepai = chepai
                max_area = contour_area
            all_chepai.append(chepai)
            # all_area.append(area)
            # cv2.imshow('chepai', chepai)


    if (real_chepai is None)|(np.size(real_chepai)==0)|rawImage.size<=(900*1100):
        real_chepai=rawImage

    return real_chepai

# path="./data/sub_test_pics/"


def prepros(path):
    exp=cv2.imdecode(np.fromfile(os.path.join("./","exp.jpg"),dtype=np.uint8),1)
    exph,expw= exp.shape[:2]
    dirlist=os.listdir(path)#这里的代码用于批量测试
    for imgname in dirlist:
        if(imgname[-4:]=='.jpg')|(imgname[-4:]=='.JPG'):
            img=cv2.imdecode(np.fromfile(os.path.join(path,imgname),dtype=np.uint8),1)
            print(imgname)
            img=locating(img,20,True)
            cv2.imwrite("../data/check/"+imgname,img)
            img=locating(img,8,False)
            # cv2.imwrite("../data/check/"+imgname,img)
            # img=cv2.resize(img,(expw,exph),interpolation=cv2.INTER_CUBIC)
            # cv2.imwrite("../data/res/"+imgname[0:-4]+".jpg", img)
            # cv2.imwrite("./data/pre_res/"+imgname, img)
            # height=img.shape[0]
            # width=img.shape[1]
            # if(height<height_min or width<width_min):#如果图像尺寸小于设定值则放大
            #     img = cv2.resize(img, (width_min, height_min), interpolation=cv2.INTER_LANCZOS4)
            #     height=height_min
            #     width=width_min
            # img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            # area_min=height*width*0.002
            # print('【导入图像】调整后的图像尺寸：',img_hsv.shape[0],'*',img_hsv.shape[1])

            # contours,rois = search_in_hsv(img_hsv)#1：在S空间中寻找可能是车牌的区域，返回区域轮廓和最小外接矩形的位置
            # rois=accurate_locate(rois)
            
            # cnt=0
            # for img_cut in rois:
            #     if img_cut.size!=0:
            #         cnt=cnt+1
            #         img_cut=cv2.resize(img_cut,(expw,exph))
                    # cv2.imwrite("./data/res/"+imgname[0:-4]+"-"+str(cnt)+".jpg", img_cut)
            if not os.path.exists("./data/res/"):
                os.makedirs("./data/res/")
            img=cv2.resize(img,(expw,exph))
            cv2.imwrite("./data/res/"+imgname,img)

            



            
# python study_opencv.py
        


    