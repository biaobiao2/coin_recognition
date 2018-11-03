# encoding:utf-8
'''
Created on 2018年11月1日

@author: liushouhua
'''

import os
import cv2
# import time
# import numpy as np
# import matplotlib.pyplot as plt

from pic_deal import show_pic


data_path = "D:\\workspace\\coin_recognition\\orbdata\\"


def testorb(img1,img2):
    """
    """
    #最大特征点数,可以修改，
#     orb = cv2.AKAZE_create()
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    
#     matches = bf.match(des1, des2)
    matches = bf.knnMatch(des1, des2,1)
    matches = [j for i in matches for j in i]
    matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, img2, flags=2)
    show_pic(img3)
    return len(matches)
    
#     img2=cv2.drawKeypoints(img2,kp2,None,(0,255,0),flags=0)

    
# def flannBMatcher(img1,img2):
#     """FlannBasedMatcher
#     """
#     sift=cv2.xfeatures2d.SIFT_create()
#     kp1,des1=sift.detectAndCompute(img1,None)
#     kp2,des2=sift.detectAndCompute(img2,None)
#      
#     FLANN_INDEX_KDTREE = 1
#     index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#     search_params = dict(checks=50)   # or pass empty dictionary
#      
#     flann = cv2.FlannBasedMatcher(index_params,search_params)
#     matches = flann.knnMatch(des1,des2,k=2)
#     matchesMask = [[0,0] for i in range(len(matches))]
#      
#     for i,(m,n) in enumerate(matches):
#         if m.distance < 0.7*n.distance:
#             matchesMask[i]=[1,0]
#              
#     draw_params = dict(matchColor = (0,255,0),singlePointColor = (255,0,0),matchesMask = matchesMask,flags = 0)
#      
#     img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
#     plt.imshow(img3,),plt.show()



def orb_deal(cropImg):
    """..\\picture\\
    """
    matchs = 0
    va = 0
    for pic in os.listdir(data_path):
        dirname = data_path + pic
        picdata = cv2.imread(dirname,0)
        match_value =  testorb(cropImg,picdata)
        if match_value >= matchs:
            matchs = match_value
            va = pic.strip().split(".")[0][-1]
    return va
            

# def test():
#     """
#     """
#     path1 = "D:\\workspace\\coin_recognition\\orbdata\\test-2-0-5.jpg"
#     path2 = "D:\\workspace\\coin_recognition\\orbdata\\test-2-1-5.jpg"
#     picdata1 = cv2.imread(path1,0)
#     picdata2 = cv2.imread(path2,0)
#     orb = cv2.ORB_create(500)
#     print picdata2,type(picdata2)
#     kp1, des1 = orb.detectAndCompute(picdata1, None)
#     kp2, des2 = orb.detectAndCompute(picdata2, None)
#     print des1
#     print des2
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     matches = bf.match(des1, des2)
#     print matches

        
        


if __name__ == "__main__":
    print "start"
    test = "test-1-1-1.jpg"

    print "finsh"
    
    
    
    
