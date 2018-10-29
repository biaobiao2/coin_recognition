# encoding:utf-8
'''
Created on 2018年10月29日

@author: liushouhua
'''

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import sys
sys.setrecursionlimit(10000)
pic_path = "coin.jpg"

def pic_show(data):
    """显示图片
    """
    cv2.imshow('ret',data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def hist():
    """直方图
    """
    img = cv2.imread(pic_path)
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#     print hsv
#     pic_show(hsv)
    color=('b','g','r')
    
    for i,col in enumerate(color):
        print i,col
        hist=cv2.calcHist([hsv],[i],None,[256],[0,256])
        print hist
#         pic_show(hist)
#     
#         plt.plot(hist)
#         
#         plt.xlim([0,256])
#         
#         plt.show()

def colorpic2gray():
    """图片二值化
    """   
    img = cv2.imread(pic_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #5×5 内核的高斯平滑
#     gray = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    #膨胀，腐蚀白色区域
    
    
    while True:
        cv2.imshow("Image", gray)
        thresh = cv2.getTrackbarPos('thres', 'Image')
        img = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)[1]
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
    
    
    img = thresh
    
    while True:
        cv2.imshow("My", img)
        # 键盘检测函数，0xFF是因为64位机器
        k = cv2.waitKey(1) & 0xFF
        if k == ord('e'):
        # 加上iterations是为了记住这个参数，不加也行
            img = cv2.erode(img, kernel, iterations=1)
            print 'erode'
        if k == ord('d'):
            img = cv2.dilate(img, kernel, iterations=1)
            print 'dilate'
        if k == ord('r'):
            img = thresh
            print 'return threshold image'
        if k == ord('q'):
            break

    cv2.destroyAllWindows()
    
    
    #边缘检测 return 所处理的图像, 轮廓的点集,轮廓的索引
    contours, hierarchy ,opt = cv2.findContours(img.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#     Perimeter=cv2.arcLength(hierarchy,True)
#     print Perimeter
    
    for cnt in hierarchy:
        M = cv2.moments(cnt)
        #计算轮廓边长
        Perimeter=cv2.arcLength(cnt,True)
        if Perimeter > 100:
            print Perimeter
#         cv2.drawContours(img,cnt,-1,(0,0,255),3) 
#     cv2.imshow("img", img)
#     cv2.waitKey(0)
    
#     pic_show(thresh) 
def on_trace_bar_changed(args):
    pass

    
if __name__ == "__main__":
    print "start"
    cv2.createTrackbar('thres', 'Image', 0, 255, on_trace_bar_changed)
#     colorpic2gray()
    thresh = cv2.getTrackbarPos('thres', 'Image')
    print "finish"

    

    
    
