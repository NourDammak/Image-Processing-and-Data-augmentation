# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 21:16:12 2022

@author: Nour Dammak
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt



def StretchingImage(image):
    """
    This function allows to:
        - Stretching an Image from 0 to 255 (normalization)
        - display the original with the Stretchedimage in one figure by matplotlib.pyplot tool
        - save the Stretched image in the current folder
             
    """
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    stretched_img = np.zeros((800,800))
    stretched_image = cv2.normalize(image, stretched_img, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    stretched_image = stretched_image.astype(np.uint8)
    stretched_imageRGB = cv2.normalize(imageRGB, stretched_img, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    stretched_imageRGB = stretched_imageRGB.astype(np.uint8)
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(imageRGB)
    plt.subplot(1, 2, 2)
    plt.title("Stretched")
    plt.imshow(stretched_imageRGB)
    plt.show()
    
    cv2.imwrite(r'StretchedImage.png',stretched_image)
    return



image = cv2.imread(r'photo_1.png')
#run the function
StretchingImage(image)