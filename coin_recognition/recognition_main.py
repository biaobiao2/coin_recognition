# encoding:utf-8
'''
Created on 2018年10月31日

@author: group
'''
import os
import cv2
import sys
import math
import time
import numpy as np
from threading import Thread
sys.path.append('../')

from coin_recognition import  pic_path, data_path
from coin_recognition.pHash_test import phash_match


def getDistanceByPosition(pointA,pointB):
    """获取直线距离
    """
    return math.sqrt(pow(pointA[0] - pointB[0],2)+pow(pointA[1] - pointB[1],2))

def getMidPoint(pointA, pointB):
    """返回线段中心点
    """
    return (int((pointA[0] + pointB[0]) * 0.5), int((pointA[1] + pointB[1]) * 0.5))

def deal_block(gray):
    """对白色区域腐蚀和膨胀，使边缘连续
    @param gray: 二值化的图像
    @return ：处理后的图像数据，格式与输入保持一致
    """
    img = gray.copy()

    while True:
        cv2.imshow("My", img)
        # 键盘检测函数，0xFF是因为64位机器
        k = cv2.waitKey(0) & 0xFF
        if k == ord('e'):
            img = cv2.erode(img, None, iterations=1)
            print 'erode'
        if k == ord('d'):
            img = cv2.dilate(img, None, iterations=1)
            print 'dilate'
        if k == ord('r'):
            img = gray
            print 'return image'
        if k == ord('q'):
            break
    return img

def recognition(standard):
    """硬币识别主函数
    @param standard: float 待处理图像最左边物体尺寸 单位:mm
    """
    img0 = cv2.imread(pic_path)
    gray = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
    #高斯矩阵的尺寸(只能取奇数)越大，标准差越大，处理过的图像模糊程度越大
#     gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    #边缘检测
    gray = cv2.Canny(gray, 50, 100)
    img = deal_block(gray)
    show_pic(img)
    
    #return 所处理的图像, 轮廓的点集,轮廓的属性矩阵
    _, cnts ,_ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    #轮廓排序，使第一个轮廓为最左边参照物
    cnts = sorted(cnts, key=lambda k : np.min(k[:,:,0]), reverse=False)  
    
    scale = None
    blue = (255, 0, 0)
    green = (0, 255, 0)
    red = (0, 0, 255)
    pink = (255, 0, 255)
    orig = img0.copy()
    #用于画物体轮廓
    orig0 = img0.copy()
    count = 0
    for c in cnts:
        if cv2.contourArea(c) < 200:
            #去除小轮廓
            continue
        
        cv2.drawContours(orig0,[c],-1,(255,255,0),2)
        show_pic(orig0)
        # 获取最小包围矩形
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect) 
        cv2.drawContours(orig, [box.astype("int")], -1, green, 2)
        show_pic(orig)
        
        for point in box:
            cv2.circle(orig, (point[0], point[1]), 5, red, -1)
            show_pic(orig)
     
        (p1, p2, p3, p4) = box
        midpoint1 = getMidPoint(p1, p2)
        midpoint2 = getMidPoint(p2, p3)
        midpoint3 = getMidPoint(p3, p4)
        midpoint4 = getMidPoint(p4, p1)
        
        for midpoint in [midpoint1,midpoint2,midpoint3,midpoint4]:
            cv2.circle(orig, midpoint, 5, blue, -1)
        show_pic(orig)
        
        cv2.line(orig, midpoint1, midpoint3,pink, 2)
        cv2.line(orig, midpoint2, midpoint4,pink, 2)
        show_pic(orig)
        
        #通过查看参照物来初始化pixelsPerMetric变量
        dis13 = getDistanceByPosition(midpoint1, midpoint3)
        dis24 = getDistanceByPosition(midpoint2, midpoint4)
        if scale is None:
            if dis24 > dis13:
                scale = dis24 / standard 
            else:
                scale = dis13 / standard 
        reald1 = dis13 / scale
        reald2 = dis24 / scale
        
        if reald1 > reald2:
            rad = reald1
        else:
            rad = reald2
         
        if float(abs(reald1-rad)) / rad > 0.1 or float(abs(reald2-rad)) / rad > 0.15 :
            #过滤不是圆形物体
            continue
        if rad >= 270 or rad <= 130:
            #过滤较小物体
            continue
        
        value = "unknown"
        if rad > 235:
            value = "1 yuan"
        else :
            Ymin = np.min(c[:,:,0])
            Ymax = np.max(c[:,:,0])
            Xmin = np.min(c[:,:,1])
            Xmax = np.max(c[:,:,1])
            orig1 = cv2.imread(pic_path)
            cropImg = orig1[Xmin:Xmax,Ymin:Ymax]
            value = orb_deal(cropImg)
            
            if value is None:
                #感知哈希
                orig1 = cv2.imread(pic_path,0)
                cropImg = orig1[Xmin:Xmax,Ymin:Ymax]
                value = phash_match(cropImg)
                
            if value is None:
                if abs(rad - 190) <= 15:
                    value =  "1 jiao"
                else:
                    value =  "5 jiao"
            else:
                value = str(value) + "jiao"
        
        #照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
        cv2.putText(orig,'%s'%(value),(midpoint1[0] - 10, midpoint1[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv2.putText(orig,'%.1fmm'%(rad/10.0),(midpoint2[0] + 10, midpoint2[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        show_pic(orig)
        count += 1
        
    cv2.putText(orig,'num of coin: %s'%(str(count)),(20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    show_pic(orig)
        
def async(f):
    def inner(*args, **kwargs):
        thr = Thread(target = f, args = args, kwargs = kwargs)
        thr.start()
    return inner

@async
def pic_show(data):
    """显示图片
    """
    cv2.namedWindow("show",0)
    cv2.resizeWindow("show", 500, 640)
    cv2.imshow('show',data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_pic(data):
    """显示图片的时间
    """
    pic_show(data)
    time.sleep(0.1)

def orb_match(img1,img2):
    """orb 特征匹配
    @param img1,img2: 格式保持一致的图片data
    @return 匹配距离小于一定值的特征数目
    """
    #最大特征点数,可以修改，
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    matches = bf.match(des1, des2)
    
    matches = sorted(matches, key=lambda x: x.distance,reverse = False)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, img2, flags=2)
    show_pic(img3)
    matches = [i for i in matches if i.distance <= 50]
    return len(matches) 


def orb_deal(img):
    """将读入图片与样本数据进行orb图像匹配，找到最相似的一张
    @param param: 
    @return: 最可能的图像硬币价值，none代表与样本数据都不匹配
    """
    matchs = 0
    for pic in os.listdir(data_path):
        dirname = data_path + pic
        picdata = cv2.imread(dirname)
        match_value =  orb_match(img,picdata)
#         break
        if match_value >= matchs:
            matchs = match_value
            matchs_value = pic.strip().split(".")[0][-1]
    if matchs > 25:
        return matchs_value
    else:
        return None

if __name__ == "__main__":
    print "start"
    #参照物半径尺寸,需要人工给出
    standard = float(sys.argv[1])
    recognition(standard)
    print "finsh"
    
    
    
    
    