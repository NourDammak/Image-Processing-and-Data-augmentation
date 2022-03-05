# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 00:57:05 2022

@author: Nour Dammak
"""
#import all functions responsible to Analysing type of Image and Data Augmenation in Image processing:
from functions import *


#Define all arguments of the functions
gray=cv2.imread(r'photo_1.png')
image = cv2.imread(r'photo_1.png')
image1 = cv2.imread(r'photo_1.png',0)
image2 = cv2.imread(r'photo_1.png',0)
OriginalImage= cv2.imread(r'photo_1.png')


"""list of all available functions"""



Analysing_The_Type_Of_Image(image)
Sharpening(image)
Flipping(image)
PyramideLevelling(image)
Rotation(image)
Resizing(image)
Cropping(image)
Rotation_Matrix2D(image)
Kernel_Functions(image)
Binary_and_GrayScale((OriginalImage,image1,image2))
Sobel_Operator(gray,image)
Fast_Algorithm(gray,image)
Sift_Keypoints(gray,image)



#we can use the function help(name_of_the_function) in order to get the docstring and read the desciption of
#each Image Processing in this module
help(Flipping)






