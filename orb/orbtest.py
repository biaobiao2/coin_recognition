# encoding:utf-8
'''
Created on 2018年11月1日

@author: liushouhua
'''

from Coin_recognition import pic_path,pic_path1

import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread(pic_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(pic_path1, cv2.IMREAD_GRAYSCALE)
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:80], img2, flags=2)

plt.imshow(img3), plt.show()
