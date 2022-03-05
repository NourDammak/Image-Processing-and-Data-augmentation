# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 15:45:09 2022

@author: Nour Dammak
"""

import cv2
import matplotlib.pyplot as plt



def Dimensions_Image(image):
    """This function allows to:
        Compute and display the dimensions : height, width and the number of channels in the image"""
    
    height, width, channels = image.shape
    print("height =",height,"/ width =", width,"/ number of channels =", channels)
    
      
def Picking_Pixels(image):
    """This function allows to:
    - Compute the dimensions of the image : height, width  and the number of channels in the image
    - Pick pixels by selecting the starting and stopping pixel for each dimension in order to crop another image
    - Display the Original image and the picked image in one figure by matplotlib.pyplot tool
    - Display the picked image in another window by the imshow tool
    - Save the picked Image in the current folder named Picked_Image.png """
    
    #display the height and the width of the image
    height, width, channels = image.shape
    print("the height =",height, "/ the width =",width)
    
    #choosing the parameters of the cropped image  while respecting the logical conditions
    
    #picking the starting height

    while 1:
        y_start=int(input("please pick the starting height :")) 
        if y_start<0:
            print("starting height must be a positiv value")
            False
            y_start=-1        
        elif y_start>=height:
            print("starting height must be strictly lower than the max height value which is :", height)
            False
            y_start=-1
        else:
            break
    
    #picking the stopping height
    
    while 1:
        y_stop=int(input("please pick the stopping height :"))
        if y_stop<0:
            print("stopping height must be a positive value")
            False
            y_stop=-1
        elif y_stop>height:
            print("stopping height must be lower or equal to the max height value which is :", height)
            False
            y_stop=-1
        elif y_stop<=y_start:
            print("stopping height must be strictly greater than the starting height value which is :", y_start)
            False
            y_stop=-1
        else:
            break
    
    #picking the starting width
    while 1:
        x_start=int(input("please pick the starting width :"))
        
        if x_start<0:
            print("starting width must be a positive value")
            False
            x_start=-1        
        elif x_start>=width:
            print("starting width must be strictly lower than the max width value which is :", width)
            False
            x_start=-1
        else:
            break
    
    #picking the stopping width
    while 1:
        x_stop=int(input("please pick the stopping width :"))
        if x_stop<0:
            print("stopping width must be a positive value")
            False
            x_stop=-1
        elif x_stop>width:
            print("stopping width must be lower or equal to the max width value which is :", width)
            False
            x_stop=-1
        elif x_stop<=x_start:
            print("stopping width must be strictly greater than the starting width value which is :", x_start)
            False
            x_stop=-1
        else:
            break
    
    
    #Processing_Picked_Image
    PI = image[y_start:y_stop, x_start:x_stop]
    #Displaying_the_picked_image
    cv2.imshow("Cropped", PI)
    #Saving Image
    cv2.imwrite(r'Cropped_Image.png',PI)
    #print the dimensions of the Picked Image
    print(" - the dimensions of the Cropped Image are presented below :") 
    Dimensions_Image(PI)
    
    #Plotting the original Image and the picked Image in one figure by matplotlib.pyplot tool                                  
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    PI_imageRGB = cv2.cvtColor(PI, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(80, 80))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(imageRGB)
    plt.subplot(1, 2, 2)
    plt.title("Cropped_Image")
    plt.imshow(PI_imageRGB)
    plt.show()





#reading the image
image=cv2.imread(r'photo_1.png')
#displaying the dimensions of the image
Dimensions_Image(image)
#Picking pixels of the image and cropping it
Picking_Pixels(image)

