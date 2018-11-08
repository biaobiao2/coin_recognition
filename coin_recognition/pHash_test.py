# encoding:utf-8
'''
Created on 2018年11月8日

@author: liushouhua
'''

"""
感知哈希算法(pHash) 相关博客:https://blog.csdn.net/sinat_26917383/article/details/70287521
"""
import os
import cv2
import sys
import math
import numpy as np

from coin_recognition import  pic_path ,data_path
from pic_deal import show_pic
from coin_recognition.orbtest import orb_deal

def pHash(img):
    """获取图片的哈希值"""
    #加载并调整图片为32x32灰度图片
#     img=cv2.imread(img, 0) 
    img=cv2.resize(img,(32,32),interpolation=cv2.INTER_CUBIC)
    vis0 = img.astype("float32")
    #二维Dct变换
    vis1 = cv2.dct(cv2.dct(vis0))
    
    size = 32  #该数的平方能被4整除
    vis1.resize(size,size)
    #把二维list变成一维list
    img_list = []
    for item in vis1.tolist():
        img_list.extend(item)
    #计算均值
    avg = float(sum(img_list))/len(img_list)
    avg_list = ['0' if i<avg else '1' for i in img_list]
    #得到哈希值
    hashstr = ""
    for x in range(0,len(avg_list),4):
        hashstr += '%x'%int(''.join(avg_list[x:x+4]),2)
    return hashstr

def hammingDist(s1, s2):
    """汉明距离
    """
    assert len(s1) == len(s2)
    dis = [ch1 != ch2 for ch1, ch2 in zip(s1, s2)]
    return sum(dis),len(dis)

def phash_match(img):
    """感知哈希算法寻找最匹配的图片
    """
    HASH0 = pHash(img)
    score = 0
    mon = ""
    for pic in os.listdir(data_path):
        va = pic.strip().split(".")[0][-1]
        HASH1 = pHash(data_path + pic)
        out_score,length = hammingDist(HASH1,HASH0)
        sco = float(out_score)/length
        if float(out_score)/length >= score:
            score = sco
            mon = va
    return mon

if __name__ == "__main__":
    print "start"
    HASH0 = pHash("test_1_1.jpg" )
    for pic in os.listdir(data_path):
        img=cv2.imread(data_path + pic, 0)
        HASH1 = pHash(img)
#         out_score = 1 - hammingDist(HASH1,HASH0)*1. / (32*32/4)
        out_score,length = hammingDist(HASH1,HASH0)
        print pic,float(out_score)/length
        
        
        
        
        

    