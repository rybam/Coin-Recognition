import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


for i in os.listdir('C:/Users/marek/OneDrive/Pulpit/studia/sem6/UM/Image/'):

    img = cv2.imread("C:/Users/marek/OneDrive/Pulpit/studia/sem6/UM/Image/" + i)
    # plt.imshow(img)
    # plt.show()
    height, width, channels = img.shape
    print(type(height), width, channels)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[int(height/3.5):int(height-height/3.5),int(width/3.5):int(width-width/3.5)]
   #  # plt.imshow(img)
   #  # plt.show()
   #  lower = [75,75,75]
   #  upper = [250, 250, 250]
   #
   #  # create NumPy arrays from the boundaries
   #  lower = np.array(lower, dtype="uint8")
   #  upper = np.array(upper, dtype="uint8")
   #  mask = cv2.inRange(img,lower,upper)
   #  mask = 255 - mask
   #  # plt.imshow(mask)
   #  # plt.show()
   #  cont,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
   #  #const_img = cv2.drawContours(img, cont,-1,255,3)
   #
   #
   #  c = max(cont,key = cv2.contourArea)
   #  x,y,w,h = cv2.boundingRect(c)
   # # cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0),5)
   #
   #  #croped_image  = cv2.resize(img[y-100:y+h,x:x+w], (300,300))
   #  croped_image = cv2.resize(img[y:y + h, x:x + w], (300, 300))
    cv2.imwrite("C:/Users/marek/OneDrive/Pulpit/studia/sem6/UM/ImageNew/"+"new"+i, img)
    # plt.imshow(img)
    # plt.show()