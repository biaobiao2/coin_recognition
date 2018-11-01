# encoding:utf-8
'''
Created on 2018年10月31日

@author: liushouhua
'''

import cv2
import time
import math
# import imutils
import numpy as np

# import matplotlib.pyplot as plt

from pic_deal import pic_show
from Coin_recognition import pic_path
#参照物尺寸
standard = 45

def show_pic(data):
    """显示图片时间
    """
    pic_show(data)
    time.sleep(0.8)

def getDistanceByPosition(pointA,pointB):
    """获取直线距离
    """
    return math.sqrt(pow(pointA[0] - pointB[0],2)+pow(pointA[1] - pointB[1],2))

def getMidPoint(pointA, pointB):
    """返回中心点
    """
    return (int((pointA[0] + pointB[0]) * 0.5), int((pointA[1] + pointB[1]) * 0.5))


def test1():
    """
    """
    img0 = cv2.imread(pic_path)
    gray = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
    #高斯矩阵的尺寸(只能取奇数)越大，标准差越大，处理过的图像模糊程度越大
#     gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    #边缘检测
    gray = cv2.Canny(gray, 50, 100)
    
    img = gray
    
#     img = cv2.dilate(img, None, iterations=1)
#     img = cv2.erode(img, None, iterations=1)

    while True:
        cv2.imshow("My", img)
        # 键盘检测函数，0xFF是因为64位机器
        k = cv2.waitKey(0) & 0xFF
        if k == ord('e'):
        # 加上iterations是为了记住这个参数，不加也行
            img = cv2.erode(img, None, iterations=1)
            print 'erode'
        if k == ord('d'):
            img = cv2.dilate(img, None, iterations=1)
            print 'dilate'
        if k == ord('r'):
            img = gray
            print 'return threshold image'
        if k == ord('q'):
            break
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
    for c in cnts:
        if cv2.contourArea(c) < 100:
            #去除小轮廓
            continue
        
        orig1 = img0.copy()
        cv2.drawContours(orig1,[c],-1,(255,255,0),2)
        orig = img0.copy()
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
        
        #照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
        cv2.putText(orig,'%.1fmm'%(reald1),(midpoint1[0] - 10, midpoint1[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        cv2.putText(orig,'%.1fmm'%(reald2),(midpoint2[0] + 10, midpoint2[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        show_pic(orig)
        show_pic(orig1)

    
    
    
if __name__ == "__main__":
    print "start"
    test1()
    print "finsh"
    
    
    
    
    