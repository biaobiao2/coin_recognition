# encoding:utf-8
'''
Created on 2018年10月31日

@author: group
'''

import cv2
import sys
import math
import numpy as np
sys.path.append('../')

from coin_recognition import  pic_path
from pic_deal import show_pic
from coin_recognition.orbtest import orb_deal

#参照物尺寸
standard = 190
standard5 = 210
# standard = 250


def getDistanceByPosition(pointA,pointB):
    """获取直线距离
    """
    return math.sqrt(pow(pointA[0] - pointB[0],2)+pow(pointA[1] - pointB[1],2))

def getMidPoint(pointA, pointB):
    """返回中心点
    """
    return (int((pointA[0] + pointB[0]) * 0.5), int((pointA[1] + pointB[1]) * 0.5))

def deal_block(gray):
    """对白色区域腐蚀和膨胀，使边缘连续
    @param gray: 二值化的图像
    @return ：处理后的图像数据，格式与输入保持一致
    """
    img = gray.copy()
    img = cv2.dilate(img, None, iterations=1)
    img = cv2.dilate(img, None, iterations=1)
    img = cv2.dilate(img, None, iterations=1)
    img = cv2.erode(img, None, iterations=1)
    img = cv2.erode(img, None, iterations=1)
    img = cv2.erode(img, None, iterations=1)
#     while True:
#         cv2.imshow("My", img)
#         # 键盘检测函数，0xFF是因为64位机器
#         k = cv2.waitKey(0) & 0xFF
#         if k == ord('e'):
#         # 加上iterations是为了记住这个参数，不加也行
#             img = cv2.erode(img, None, iterations=1)
#             print 'erode'
#         if k == ord('d'):
#             img = cv2.dilate(img, None, iterations=1)
#             print 'dilate'
#         if k == ord('r'):
#             img = gray
#             print 'return threshold image'
#         if k == ord('q'):
#             break
    return img

def recognition():
    """
    """
    img0 = cv2.imread(pic_path)
    gray = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
    #高斯矩阵的尺寸(只能取奇数)越大，标准差越大，处理过的图像模糊程度越大
#     gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    #边缘检测
    gray = cv2.Canny(gray, 50, 100)
    img = deal_block(gray)
#     show_pic(img)
    
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
    count = 0
    for c in cnts:
        if cv2.contourArea(c) < 100:
            #去除小轮廓
            continue
        count += 1
        orig1 = img0.copy()
        cv2.drawContours(orig1,[c],-1,(255,255,0),2)
#         show_pic(orig1)
        # 获取最小包围矩形
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect) 
        cv2.drawContours(orig, [box.astype("int")], -1, green, 2)
#         show_pic(orig)
        
        for point in box:
            cv2.circle(orig, (point[0], point[1]), 5, red, -1)
#             show_pic(orig)
     
        (p1, p2, p3, p4) = box
        midpoint1 = getMidPoint(p1, p2)
        midpoint2 = getMidPoint(p2, p3)
        midpoint3 = getMidPoint(p3, p4)
        midpoint4 = getMidPoint(p4, p1)
        
        for midpoint in [midpoint1,midpoint2,midpoint3,midpoint4]:
            cv2.circle(orig, midpoint, 5, blue, -1)
#         show_pic(orig)
        
        cv2.line(orig, midpoint1, midpoint3,pink, 2)
        cv2.line(orig, midpoint2, midpoint4,pink, 2)
#         show_pic(orig)
        
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
        
        if float(abs(reald1-rad)) / rad > 0.1 or float(abs(reald2-rad)) / rad > 0.1 :
            #过滤不是圆形物体
            continue
        if rad >= 260 or 235 <= 150:
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
            if value == -1:
                if abs(rad - 190) <= 13:
                    value =  "1 jiao"
                else:
                    value =  "5 jiao"
            else:
                value = str(value) + "jiao"
        
        #照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
        cv2.putText(orig,'%s'%(value),(midpoint1[0] - 10, midpoint1[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        cv2.putText(orig,'%.1fmm'%(rad/10.0),(midpoint2[0] + 10, midpoint2[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        show_pic(orig)
        
    cv2.putText(orig,'num of coin: %s'%(str(count)),(20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 4)
    show_pic(orig)
        

    
if __name__ == "__main__":
    print "start"
    recognition()
    print "finsh"
    
    
    
    
    