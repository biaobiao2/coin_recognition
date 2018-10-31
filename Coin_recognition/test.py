# encoding:utf-8
'''
Created on 2018年10月31日

@author: liushouhua
'''

import cv2
import imutils
import numpy as np
from imutils import perspective
from scipy.spatial import distance as dist
import time

from pic_deal import pic_show

pic_path = "coin.jpg"

def midpoint(ptA, ptB):
    """返回中心点
    """
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def test1():
    """
    """
    img0 = cv2.imread(pic_path)
    gray = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
    #高斯矩阵的尺寸(只能取奇数)越大，标准差越大，处理过的图像模糊程度越大
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    #边缘检测
    gray = cv2.Canny(gray, 50, 100)
#     
    img = gray
#     
    img = cv2.dilate(img, None, iterations=1)
    img = cv2.erode(img, None, iterations=1)
    pic_show(img)
#     pic_show(img)
#     #dilate:膨胀，erode:腐蚀白色区域
#     while True:
#         cv2.imshow("My", img)
#         # 键盘检测函数，0xFF是因为64位机器
#         k = cv2.waitKey(1) & 0xFF
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
    
    #return 所处理的图像, 轮廓的点集,轮廓的索引
    _, cnts ,_ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    (cnts, _) = imutils.contours.sort_contours(cnts)
    pixelsPerMetric = None
    
    i = 0
    
    for c in cnts:
        if cv2.contourArea(c) < 100:
            #去除小轮廓
            continue
        orig = img0.copy()
        # 获取最小包围矩形
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box) 
        box = np.array(box, dtype="int")
        #矩形点透视
        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
        
        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
#             cv2.imshow("My", orig)
     
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)
        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255, 0, 255), 2)
#         pic_show(orig)
        
        #通过查看参照物来初始化pixelsPerMetric变量
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        if pixelsPerMetric is None:
            pixelsPerMetric = dB / 45
        
        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric
        
        cv2.putText(orig, "{:.1f}in".format(dimA),(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(orig, "{:.1f}in".format(dimB),(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

#         cv2.imshow("Image", orig)
#         cv2.waitKey(0)
        
        cv2.imwrite("cat2_%s.jpg"%i, orig)
        i += 1

    
    
    
if __name__ == "__main__":
    print "start"
    test1()
    print "finsh"