# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 12:11:05 2022

@author: Nour Dammak
"""

import cv2


def MouseCroppingFunction(event, x, y, flags, param):
    """this function allows to :
        - crop Image from an original image in rectangle shape using mouse left button click
        - print the coordinates of the image in the Python console
        - save the coordinates of the cropped image in a txt file named Cropped_Image_Coordinates
        - display the cropped image in an independant window
        - save the cropped image in the current folder named CroppedImage.png """ 
    
    global rectanglePoints, crop
    if event == cv2.EVENT_LBUTTONDOWN:
        crop = True
        rectanglePoints = [ (x,y),(x,y) ]
        print("x0 =",x," /y0 =",y)
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if crop == True:
            rectanglePoints[1]=x, y
            
    elif event == cv2.EVENT_LBUTTONUP:
        rectanglePoints[1]=x,y

        y0 = rectanglePoints[0][1]
        y1 = rectanglePoints[1][1]
        x0 = rectanglePoints[0][0]
        x1 = rectanglePoints[1][0]
        crop = False  
        print(rectanglePoints)
        
        if len(rectanglePoints) == 2: 
            if x0<x1 and y0<y1:
                CroppedImage = img[y0:y1, x0:x1]
            elif x1<x0 and y0<y1:
                CroppedImage = img[y0:y1, x1:x0]           
            elif x0<x1 and y1<y0:
                CroppedImage = img[y1:y0, x0:x1]
            else:
                CroppedImage = img[y1:y0, x1:x0]
            file = open("Cropped_Image_Coordinates.txt","w")
            file.write('Y0[pix],Y1[pix],X0[pix],X1[pix]\n')
            file.write('%f,%f,%f, %f\n' %(y0,y1,x0,x1))
            file.close()    
            cv2.imshow("MouseCropped", CroppedImage)
            cv2.imwrite('MouseCroppedImage.jpg', CroppedImage)
            


img=cv2.imread(r"photo_1.png")
crop = False
rectanglePoints0 = []

cv2.namedWindow("image")
cv2.setMouseCallback("image", MouseCroppingFunction)
final_img=cv2.imread(r"photo_1.png")

while 1:

    Img = final_img.copy()
    if crop==False:
        cv2.imshow("image", Img)
    elif crop==True:
        cv2.rectangle(Img, (rectanglePoints[0][0], rectanglePoints[0][1]), (rectanglePoints[1][0], rectanglePoints[1][1]), (0, 0, 255), 2)
        cv2.imshow("image", Img)   
    cv2.waitKey(1)
    if cv2.waitKey(20) & 0xFF ==27: 
        break
    
cv2.destroyAllWindows()
