# encoding:utf-8
'''
Created on 2018年10月29日

@author: liushouhua
'''

import cv2
from threading import Thread
# from PIL import Image
# import matplotlib.pyplot as plt

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

def hist():
    """直方图
    """
    img = cv2.imread(pic_path)
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    color=('b','g','r')
    
    for i,_ in enumerate(color):
        hist=cv2.calcHist([hsv],[i],None,[256],[0,256])
#         plt.plot(hist)
#         plt.xlim([0,256])
#         plt.show()

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
    colorpic2gray()

    print "finish"

    

    
    
