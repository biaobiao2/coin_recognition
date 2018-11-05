# encoding:utf-8
'''
Created on 2018年10月29日

@author: group
'''
import os
import cv2
import time
import numpy as np
from threading import Thread
# from PIL import Image
import matplotlib.pyplot as plt

from coin_recognition import data_path
pic_path = "coin.jpg"

def async(f):
    def inner(*args, **kwargs):
        thr = Thread(target = f, args = args, kwargs = kwargs)
        thr.start()
    return inner

@async
def pic_show(data):
    """显示图片
    """
    cv2.imshow('ret',data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_pic(data):
    """显示图片的时间
    """
    pic_show(data)
    time.sleep(1)
    

def hist(img2):
    """直方图
    """
    value = 1000
    pic_name = 0
    hist_item1 = cv2.calcHist([img2],[0],None,[256],[0,255])
    cv2.normalize(hist_item1,hist_item1,0,255,cv2.NORM_MINMAX)
    for pic in os.listdir(data_path):
        dirname = data_path + pic        
        img1 = cv2.imread(dirname,0)
        hist_item2 = cv2.calcHist([img1],[0],None,[256],[0,255])
        cv2.normalize(hist_item2,hist_item2,0,255,cv2.NORM_MINMAX)
        sc= cv2.compareHist(hist_item1, hist_item2, 0)
        if sc <= value:
            value = sc
            pic_name = pic.strip().split(".")[0].split("-")[-1]
        print sc
        print pic
        print np.array(img1.ravel()).mean()
    return pic_name

#         color = [ (255,0,0),(0,255,0),(0,0,255) ]
#         for ch, col in enumerate(color):
#             hist_item1 = cv2.calcHist([img1],[ch],None,[256],[0,255])
#             hist_item2 = cv2.calcHist([img2],[ch],None,[256],[0,255])
#             cv2.normalize(hist_item1,hist_item1,0,255,cv2.NORM_MINMAX)
#             cv2.normalize(hist_item2,hist_item2,0,255,cv2.NORM_MINMAX)
#             sc= cv2.compareHist(hist_item1, hist_item2, 0)
#             print sc
#         print pic    
    
def test():
    for pic in os.listdir(data_path):
        print pic
        dirname = data_path + pic        
        img = cv2.imread(dirname)
        hsv=cv2.cvtColor(img,0)
        color=('b','g','r')
        for i,col in enumerate(color):
            hist=cv2.calcHist([hsv],[i],None,[256],[0,256])
            plt.plot(hist)
            plt.xlim([0,256])
            plt.show()

def colorpic2gray():
    """图片二值化，对白色区域处理成块，计算轮廓个数，既硬币数
    @return: 硬币数目
    """   
    img = cv2.imread(pic_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #二值化
    _, gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #边缘检测
    thresh = cv2.Canny(gray, 50, 100)
    #dilate:膨胀，erode:腐蚀白色区域
    img = thresh
#     会返回指定形状和尺寸的结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
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
    _, hierarchy ,_ = cv2.findContours(img.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return len(hierarchy)


    
if __name__ == "__main__":
    print "start"
    test()
    print "finish"

    

    
    
