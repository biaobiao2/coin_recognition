# encoding:utf-8
'''
Created on 2018年10月29日

@author: liushouhua
'''

import numpy as np
import cv2

def getPoint(x,y,data,subdata=None):
    """泛洪填充算法
    """
    a=[0,-1,0,1,0,-2,0,2,0,-3,0,3,0,-4,0,4,0,-5,0,5]
    b=[1,0,-1,0,2,0,-2,0,3,0,-3,0,4,0,-4,0,5,0,-5,0]
    width,height=data.shape
    if subdata is None:
        subdata=[]
    if x>5 and y<height-5 and y>5 and x<width-5:
        for i in range(20):
            if data[x+a[i]][y+b[i]]==0:
                subdata.append((x+a[i],y+b[i]))
                data[x+a[i]][y+b[i]]=2
                getPoint(x+a[i],y+b[i],data,subdata)
    subdata.append((x,y))
 
def getcell(filename):
    """切割图片
    """
    data = cv2.imread(filename,2) 
#     print data
    data = data / 200 #降噪
    list1=[]
    index=0
    flag=True
    for y in range(data.shape[1]):
        for x in range(data.shape[0]):
            if data[x][y]==0:
                if list1:
                    for i in range(len(list1)):
                        if (x,y) in list1[i]:
                            flag=False
                if not flag:
                    continue
                list1.append([])
                getPoint(x,y,data,list1[index])#调用流水算法
                index+=1
            else :
                continue
    count = 0
    if len(list1) != 4:
        print len(list1)
    for lis in list1:
        left=lis[0][0]
        dwon=lis[0][1]
        right=lis[0][0]
        top=lis[0][1]
        for i in lis:
            x=i[0]
            y=i[1]
            left=min(left,x)
            dwon=min(dwon,y)
            right=max(right,x)
            top=max(top,y)
        w=right-left+1
        h=top-dwon+1
        if (w*h <8):#去除小色块
            continue
        img0=np.zeros([w,h])#创建全0矩阵
        for x,y in lis:
            img0[x-left][y-dwon]=1
        img0[img0<1]=255
        img1=Image.fromarray(img0)
        img1=img1.convert('RGB')
        img1.save(str(filename.split(".")[0])+'_2_'+str(count)+'.png')
        count += 1