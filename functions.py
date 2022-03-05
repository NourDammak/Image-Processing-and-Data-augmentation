# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 15:11:32 2022

@author: Nour Dammak
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def Analysing_The_Type_Of_Image(image):
    """
this function allows to analyse the type of the image by showing:
    - the type if it is grayscale or RGB
    - the width
    - the height
    - the number of channels
    """
        
    data =image.shape
    if len(data) ==2:
        print('the imported image is an grayscale')
        height = data[0]
        width=data[1]
        numberOfChanels = None
        return height, width, numberOfChanels
    elif len(data) ==3:
        print('The imported image is RGB')
        height = data[0]
        width=data[1]
        numberOfChanels = data[2]
        return height, width, numberOfChanels
        
    else:
        print('This is not the image')
        return None



def Sharpening(image):  
    """
This function allows to:
    - Create a sharpen Image with the kernel sharpening  = ([[-1,-1,-1], 
                              [-1,9,-1], 
                              [-1,-1,-1]])
    - display the original with the sharpen image in one figure by matplotlib.pyplot tool
    - save the sharpen images in the current folder
    
    """
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(imageRGB)
    kernel_sharpening = np.array([[-1,-1,-1], 
                                  [-1,9,-1], 
                                  [-1,-1,-1]])
    sharpened2 = cv2.filter2D(image, -1, kernel_sharpening)
    sharpened = cv2.filter2D(imageRGB, -1, kernel_sharpening)
    plt.subplot(1, 2, 2)
    plt.title("Image Sharpening")
    plt.imshow(sharpened)
    plt.show()
    cv2.imwrite(r'imageSharpening.png',sharpened2)
    return 

def Flipping(image): 
    """ 
This function allows to:
    - flip the image vertically, hrizentally and both of them (vertically and horizontally) 
    - display the original with the flipped images in one figure by matplotlib.pyplot tool
    - save the flipped images in the current folder
    """
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    flipVertical = cv2.flip(image, 0)
    flipHorizontal = cv2.flip(image, 1)
    flipBoth = cv2.flip(image, -1)
    imageVertRGB = cv2.cvtColor(flipVertical, cv2.COLOR_BGR2RGB)
    imageHorRGB = cv2.cvtColor(flipHorizontal, cv2.COLOR_BGR2RGB)
    imageBothRGB = cv2.cvtColor(flipBoth, cv2.COLOR_BGR2RGB)
    #flipVerticalRGB = cv2.flip(imageRGB, 0)
    #flipHorizontalRGB = cv2.flip(imageRGB, 1)
    #flipBothRGB = cv2.flip(imageRGB, -1)
    plt.figure(figsize=(80, 80))
    plt.subplot(2, 2, 1)
    plt.title("Original")
    plt.imshow(imageRGB)
    plt.subplot(2, 2, 2)
    plt.title("Vertical_Flipping")
    plt.imshow(imageVertRGB)
    plt.subplot(2, 2, 3)
    plt.title("Horizental_Flipping")
    plt.imshow(imageHorRGB)
    plt.subplot(2, 2, 4)
    plt.title("Both_Flipping")
    plt.imshow(imageBothRGB)
    plt.show()
    cv2.imwrite(r'Flipped_Vertical_Image.png',flipVertical)
    cv2.imwrite(r'Flipped_Horizental_Image.png',flipHorizontal)
    cv2.imwrite(r'Flipped_Both_Image.png',flipBoth)
    return 
 
def PyramideLevelling(image):
    
    """
 This function allows to:
    - create a set of images with different resolutions called Image Pyramids by 
    removing consecutive rows and columns in Lower level which called Gaussian Pyramide
    - display the original with the Image Pyramids in one figure by matplotlib.pyplot tool
    - save the Image Pyramids in the current folder
    
    """
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imagePyramideLevelOne= cv2.pyrDown(image)
    imagePyramideLevelTwo= cv2.pyrDown(imagePyramideLevelOne)
    imagePyramideLevelThree= cv2.pyrDown(imagePyramideLevelTwo)
    imagePyramideLevelFour= cv2.pyrDown(imagePyramideLevelThree)
    imagePyramideLevelFive= cv2.pyrDown(imagePyramideLevelFour)
    
    imagePyramideLevelOneRGB= cv2.pyrDown(imageRGB)
    imagePyramideLevelTwoRGB= cv2.pyrDown(imagePyramideLevelOneRGB)
    imagePyramideLevelThreeRGB= cv2.pyrDown(imagePyramideLevelTwoRGB)
    imagePyramideLevelFourRGB= cv2.pyrDown(imagePyramideLevelThreeRGB)
    imagePyramideLevelFiveRGB= cv2.pyrDown(imagePyramideLevelFourRGB)

    plt.figure(figsize=(80, 80))
    plt.subplot(2, 3, 1)
    plt.title("Original")
    plt.imshow(imageRGB)
    plt.subplot(2, 3, 2)
    plt.title("PyramideLevel_1")
    plt.imshow(imagePyramideLevelOneRGB)
    plt.subplot(2, 3, 3)
    plt.title("PyramideLevel_2")
    plt.imshow(imagePyramideLevelTwoRGB)
    plt.subplot(2, 3, 4)
    plt.title("PyramideLevel_3")
    plt.imshow(imagePyramideLevelThreeRGB)
    plt.subplot(2, 3, 5)
    plt.title("PyramideLevel_4")
    plt.imshow(imagePyramideLevelFourRGB)
    plt.subplot(2, 3, 6)
    plt.title("PyramideLevel_5")
    plt.imshow(imagePyramideLevelFiveRGB)
    plt.show()
    
    cv2.imwrite(r'imagePyramideLevel_1.png',imagePyramideLevelOne)
    cv2.imwrite(r'imagePyramideLevel_2.png',imagePyramideLevelTwo)
    cv2.imwrite(r'imagePyramideLevel_3.png',imagePyramideLevelThree)
    cv2.imwrite(r'imagePyramideLevel_4.png',imagePyramideLevelFour)
    cv2.imwrite(r'imagePyramideLevel_5.png',imagePyramideLevelFive)
    return



def Rotation(image):
    """
This function allows to:
    - rotate an image by 180, 90 and -90 degrees
    - display the original and the rotated images in one plot by matplotlib.pyplot tool
    - save the Image Pyramids in the current folder
    
    """
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    imageRotated2 = cv2.rotate(image,cv2.ROTATE_180)
    imageRotated3 = cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)  
    imageRotated4 = cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    imageRotated2RGB = cv2.rotate(imageRGB,cv2.ROTATE_180)
    imageRotated3RGB = cv2.rotate(imageRGB,cv2.ROTATE_90_CLOCKWISE)  
    imageRotated4RGB = cv2.rotate(imageRGB,cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    plt.figure(figsize=(80, 80))
    plt.subplot(2, 2, 1)
    plt.title("Original")
    plt.imshow(imageRGB)
    plt.subplot(2, 2, 2)
    plt.title("Rotated180deg")
    plt.imshow(imageRotated2RGB)
    plt.subplot(2, 2, 3)
    plt.title("Rotated-90deg")
    plt.imshow(imageRotated3RGB)
    plt.subplot(2, 2, 4)
    plt.title("Rotated90deg")
    plt.imshow(imageRotated4RGB)
    plt.show()
    
    cv2.imwrite(r'ImageRotated180deg.png',imageRotated2)
    cv2.imwrite(r'ImageRotated-90deg.png',imageRotated3)
    cv2.imwrite(r'ImageRotated90deg.png',imageRotated4)
  
    return

def Resizing(image):
    """ 
This function allows to:
    - changing the dimensions of the image
    - display the original with the resized image in one figure by matplotlib.pyplot tool
    - save the resized images in the current folder
    """
    
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    imageResizeFirstM= cv2.resize(image,None, fx=2, fy=5, interpolation=cv2.INTER_CUBIC)
    imageResizeSecondM = cv2.resize(image,(3000,1500))
    imageResizeFirstM_RGB= cv2.resize(imageRGB,None, fx=2, fy=5, interpolation=cv2.INTER_CUBIC)
    imageResizeSecondM_RGB = cv2.resize(imageRGB,(3000,1500))
    
    plt.figure(figsize=(80, 80))
    plt.subplot(2, 2, 1)
    plt.title("Original")
    plt.imshow(imageRGB)
    plt.subplot(2, 2, 2)
    plt.title("resizedM1")
    plt.imshow(imageResizeFirstM_RGB)
    plt.subplot(2, 2, 3)
    plt.title("resizedM2")
    plt.imshow(imageResizeSecondM_RGB)
    plt.show()
    
    cv2.imwrite(r'resizedImageM1.png',imageResizeFirstM)
    cv2.imwrite(r'resizedImageM2.png',imageResizeSecondM)
    return 

def Cropping(image):
    
    """
This function allows to:
    - Crop an image by selecting the starting and stopping pixel for each dimension in order to crop another images
    - display the original with the cropped images in one figure by matplotlib.pyplot tool
    - save the cropped images in the current folder
    """
    
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #Aigsize=(20, 20)
    hgt, wdt = image.shape[:2]
    hgt, wdt = imageRGB.shape[:2]
    start_row1, start_col1 = int(hgt * .25), int(wdt * .25)
    end_row1, end_col1 = int(hgt * .75), int(wdt * .75)
    start_row2, start_col2 = int(hgt * .35), int(wdt * .35)
    end_row2, end_col2 = int(hgt * .85), int(wdt * .85)
    start_row3, start_col3 = int(hgt * .10), int(wdt * .10)
    end_row3, end_col3 = int(hgt * .40), int(wdt * .40)
    cropped1 = image[start_row1:end_row1 , start_col1:end_col1] 
    croppedRGB1 = imageRGB[start_row1:end_row1 , start_col1:end_col1]
    cropped2 = image[start_row2:end_row2 , start_col2:end_col2] 
    croppedRGB2 = imageRGB[start_row2:end_row2 , start_col2:end_col2]
    cropped3 = image[start_row3:end_row3 , start_col3:end_col3] 
    croppedRGB3 = imageRGB[start_row3:end_row3 , start_col3:end_col3]
    
    plt.figure(figsize=(80, 80))
    plt.subplot(2, 2, 1)
    plt.title("Original")
    plt.imshow(imageRGB)
    plt.subplot(2, 2, 2)
    plt.title("cropped_1")
    plt.imshow(croppedRGB1)
    plt.subplot(2, 2, 3)
    plt.title("cropped_2")
    plt.imshow(croppedRGB2)
    plt.show()
    plt.subplot(2, 2, 4)
    plt.title("cropped_3")
    plt.imshow(croppedRGB3)
    plt.show()
    
    
    cv2.imwrite(r'cropped_image1.png',cropped1)   
    cv2.imwrite(r'cropped_image2.png',cropped2)
    cv2.imwrite(r'cropped_image3.png',cropped3)
    
    return
    

    

def Rotation_Matrix2D(image):

    """
This function allows to 

    - rotate an image with taking 3 arguments:
        
        - center: the center of rotation for the input image
        - angle: the angle of rotation in degrees
        - scale: an isotropic scale factor which scales the image up or down according to the value provided
        
    - display the original with the rotated image in one figure by matplotlib.pyplot tool
    - save the rotated images in the current folder 
    
    
    """
     
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x=1000
    y=500
    fi=130
    scale=3
    M=cv2.getRotationMatrix2D((x,y),fi,scale)
    rotatedImage = cv2.warpAffine(image,M,(3000,4000))
    rotatedImageRGB = cv2.warpAffine(imageRGB,M,(3000,4000))
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(imageRGB)
    plt.subplot(1, 2, 2)
    plt.title("Rotated Matrix 2D")
    plt.imshow(rotatedImageRGB )
    plt.show()
    
    cv2.imwrite(r'RotatedMatrix2DImage.png',rotatedImage)

    return

def Sharpen_Kernel(sharpenForce):
    """ Assign the Sharpen Kernel Function """
    kernel = np.array([[0, (-1*sharpenForce),0],[(-1*sharpenForce),(4*sharpenForce)+1,(-1*sharpenForce)],[0,(-1*sharpenForce),0]])
    return kernel


def Line_Kernels():
    """ Assign the Line Kernel Function """
    devX = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    devY = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    return devX, devY



def Kernel_Functions(image):
    
    """ 

 This funciton allows to 
    -create different type of image based on Kernel functions, such as:
            - Sharpen Image: using Shrpenkernel function
            - Blurred image: used for removing noise and obtained by convolving the original grayscale images with Gaussian kernels
            - MedianBlurred image : used for removing salt-and-pepper noise
            - GaussianBlurred image: used as a low-pass filter that removes the high-frequency components are reduced
            - Difference of Gaussian (DoG) filtered image: a spatial band-pass filter that attenuates frequencies in the original grayscale image
            - Gradient Magnitude image :  measure how strong the change in image intensity is.
            - Gradient Magnitude of the DoG image : measure how strong the change in the DoG_image intensity is.
    - display the original with the Kernel images in one figure by matplotlib.pyplot tool
    - save the Kernel images in the current folder
     
    """
    
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    SharpenkernelOne = Sharpen_Kernel(1)
    imageSharpen=cv2.filter2D(image,-1,SharpenkernelOne)
    imageSharpenRGB=cv2.filter2D(imageRGB,-1,SharpenkernelOne)
    
    imageBlurSimple = cv2.blur(image,(7,7))
    imageBlurSimpleRGB = cv2.blur(imageRGB,(7,7))

    imageMedianfilter = cv2.medianBlur(image,9)
    imageMedianfilterRGB = cv2.medianBlur(imageRGB,9)
    
    
    imageGaussianBlur=cv2.GaussianBlur(image, (99,99),0)
    imageGaussianBlurRGB=cv2.GaussianBlur(imageRGB, (99,99),0)
    
    
    rawImage=cv2.GaussianBlur(image,(5,5),0.5)   
    rawImageRGB=cv2.GaussianBlur(imageRGB,(5,5),0.5)  
    secondImage=cv2.GaussianBlur(image,(5,5),3)
    secondImageRGB=cv2.GaussianBlur(imageRGB,(5,5),3)
    DoG = secondImage - rawImage
    DoGRGB = secondImageRGB - rawImageRGB
    
    
    devX, devY = Line_Kernels()
    imageDevX = cv2.filter2D(image,-1,devX)
    imageDevY = cv2.filter2D(image,-1,devY)
    imageDevXRGB = cv2.filter2D(imageRGB,-1,devX)
    imageDevYRGB = cv2.filter2D(imageRGB,-1,devY)
    imageDevX_DoG = cv2.filter2D(DoG,-1,devX)
    imageDevY_DoG = cv2.filter2D(DoG,-1,devY)
    imageDevX_DoGRGB = cv2.filter2D(DoGRGB,-1,devX)
    imageDevY_DoGRGB = cv2.filter2D(DoGRGB,-1,devY)
    
    gradientMagnitude = cv2.addWeighted(imageDevX,0.5,imageDevY,0.5,0)
    gradientMagnitudeRGB = cv2.addWeighted(imageDevXRGB,0.5,imageDevYRGB,0.5,0)
    gradientMagnitudeDoG = cv2.addWeighted(imageDevX_DoG,0.5,imageDevY_DoG,0.5,0)
    gradientMagnitudeDoGRGB = cv2.addWeighted(imageDevX_DoGRGB,0.5,imageDevY_DoGRGB,0.5,0)
    
    plt.figure(figsize=(80, 80))
    plt.subplot(3, 3, 1)
    plt.title("Original")
    plt.imshow(imageRGB)
    plt.subplot(3, 3, 2)
    plt.title("Sharpen")
    plt.imshow(imageSharpenRGB)
    plt.subplot(3, 3, 3)
    plt.title("Blur")
    plt.imshow(imageBlurSimpleRGB)
    plt.subplot(3, 3, 4)
    plt.title("Medianblur_9")
    plt.imshow(imageMedianfilterRGB)
    plt.subplot(3, 3, 5)
    plt.title("Gaussianblur_99,99")
    plt.imshow(imageGaussianBlurRGB)
    plt.subplot(3, 3, 6)
    plt.title("DoG_Image")
    plt.imshow(DoGRGB)
    plt.subplot(3, 3, 7)
    plt.title("Gradient_Magnitude")
    plt.imshow(gradientMagnitudeRGB)
    plt.subplot(3, 3, 8)
    plt.title("Gradient_Magnitude_DoG_Image")
    plt.imshow(gradientMagnitudeDoGRGB)
    plt.show()

    
    cv2.imwrite(r'imageSharpen.jpg',imageSharpen) 
    cv2.imwrite(r'BlurSimple.jpg',imageBlurSimple)
    cv2.imwrite(r'MedianBlur_9.jpg',imageMedianfilter)
    cv2.imwrite(r'GaussianBlur_99,99.jpg',imageGaussianBlur)
    cv2.imwrite(r'DoG_Image.jpg',DoG)
    cv2.imwrite(r'Gradient_Magnitude.jpg',gradientMagnitude)
    cv2.imwrite(r'Gradient_Magnitude_DoG_Image.jpg',gradientMagnitudeDoG)
    return



def Binary_and_GrayScale(OriginalImage,image1,image2):
    
    """
This function allows to 
    - create a Binary image from an input image
    - create a Gray Scale image from an input image
    - display the original with the Binary and the Gray Scale images in one figure by matplotlib.pyplot tool
    - save the Binary and the Gray Scale images in the current folder
    
    """
    
    height , width , _  = Analysing_The_Type_Of_Image(image1)
    height , width , _  = Analysing_The_Type_Of_Image(image2)
    imageRGB = cv2.cvtColor(OriginalImage, cv2.COLOR_BGR2RGB)
    for i in range(height):
        for j in range(width):
            if image1[i,j] > 150:
                image1[i,j] = 255
            else:
                image1[i,j] = 0    
    for m in range(height):
        for n in range(width):
            image2[m,n] = image2[m,n] + 50
    brightnessParametr = np.ones(image2.shape, dtype ='uint8')*50
    newImage= cv2.add(image2,brightnessParametr)
    newImage=cv2.add(image2,np.ones(image2.shape,dtype ='uint8')*50)
                
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(imageRGB)
    plt.subplot(1, 3, 2)
    plt.title("Binary")
    imageRGB1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    plt.imshow(imageRGB1)
    plt.subplot(1, 3, 3)
    plt.title("GrayScale")
    imageRGB2 = cv2.cvtColor(newImage, cv2.COLOR_BGR2RGB)
    plt.imshow(imageRGB2)
    plt.show()
    cv2.imwrite(r'BinaryImage.png',image1)
    cv2.imwrite(r'GrayScaleImage.png',newImage)
    return


def Sobel_Operator(gray,image):
    
    """ 
This function allows to:
    - create a Sobel filtered image shows how abruptly or smoothly the image changes at each pixel
    - display the original with Sobel images in one figure by matplotlib.pyplot tool
    - save the Sobel images in the current folder
    
    """
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    depth=cv2.CV_16S
    gradientsX=cv2.Sobel(gray,depth,1,0)
    gradientsY=cv2.Sobel(gray,depth,0,1)
    gradientsX=cv2.convertScaleAbs(gradientsX)
    gradientsY=cv2.convertScaleAbs(gradientsY)
    magnitude=cv2.addWeighted(gradientsX,0.5,gradientsY,0.5,0)
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(imageRGB)
    plt.subplot(1, 2, 2)
    plt.title("Sobel_Magnitude")
    MagnitudeRGB=cv2.cvtColor(magnitude, cv2.COLOR_BGR2RGB)
    plt.imshow(MagnitudeRGB)
    plt.show()
    cv2.imwrite(r'Sobel_Magnitude_Image.png',magnitude)
    return




def Fast_Algorithm(gray,image):
    
    """
    
this function allows to:
    - create an image with fast key points (by an algorithm for corner detection)
    - display the original with FastKeyPoint images in one figure by matplotlib.pyplot tool
    - save the FastKeyPoint images in the current folder
    
    """

    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fast = cv2.FastFeatureDetector_create()
    #fast.setNonmaxSuppression(0)
    fastKeyPoint = fast.detect(gray,None)
    drawFastKeypoints = cv2.drawKeypoints(image,fastKeyPoint,None,color = (255,0,0)) 
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(imageRGB)
    plt.subplot(1, 2, 2)
    plt.title("Fast Key Points")
    drawFastKeypointsRGB=cv2.cvtColor(drawFastKeypoints, cv2.COLOR_BGR2RGB)
    plt.imshow(drawFastKeypointsRGB)
    plt.show()
    cv2.imwrite(r'Fast_Key_Points.png',drawFastKeypoints)
    return


def Sift_Keypoints(gray,image):
    
    """
    
This function allows to:
    - create a SIFT image (Scale-Invariant Feature Transform)
    - display the original with SIFT images in one figure by matplotlib.pyplot tool
    - save the SIFT images in the current folder       
        
    """
    
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sift = cv2.SIFT_create()
    siftKeyPoints=sift.detect(gray,None)
    drawSiftPoints=cv2.drawKeypoints(image,siftKeyPoints,image,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(imageRGB)
    plt.subplot(1, 2, 2)
    plt.title("Sift_Key_Points")
    drawSiftPointsRGB=cv2.cvtColor(drawSiftPoints, cv2.COLOR_BGR2RGB)
    plt.imshow(drawSiftPointsRGB)
    plt.show()  
    cv2.imwrite(r'Sift_Key_Points.png',drawSiftPoints)
    return
                
                




