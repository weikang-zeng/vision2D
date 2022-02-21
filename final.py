#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 10:32:04 2021

@author: zeng
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 11:29:51 2021

@author: zeng
"""

import cv2
import numpy as np


lower_blue = np.array([110,150,50])
upper_blue = np.array([120,255,255])
lower_lightviolet = np.array([127,82,190])
upper_lightviolet = np.array([133,113,245])
lower_green = np.array([25,58,116])
upper_green = np.array([38,114,177])
lower_red = np.array([165,142,147])
upper_red = np.array([180,168,255])
lower_yellow = np.array([8,74,136])
upper_yellow = np.array([24,120,255])
lower_violet = np.array([127,119,126])
upper_violet = np.array([134,151,255])
lower_redviolet = np.array([155,217,116])
upper_redviolet = np.array([161,245,232])



s1=0
s2=1
s3=2
s4=3




cam = cv2.VideoCapture('paperEval.mp4') 

if (not cam.isOpened):
    print ('Impossible to camera !')


while(True):
    ret, img = cam.read()  
       

# Convert BGR into HSV and display
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_b = cv2.inRange(hsv,lower_blue,upper_blue)
    output_b = cv2.bitwise_and(img,img,mask=mask_b)

    mask_lv = cv2.inRange(hsv,lower_lightviolet,upper_lightviolet)
    output_lv = cv2.bitwise_and(img,img,mask=mask_lv)

    mask_r = cv2.inRange(hsv,lower_red,upper_red)
    output_r = cv2.bitwise_and(img,img,mask=mask_r)

    mask_g = cv2.inRange(hsv,lower_green,upper_green)
    output_g = cv2.bitwise_and(img,img,mask=mask_g)

    mask_y = cv2.inRange(hsv,lower_yellow,upper_yellow)
    output_y = cv2.bitwise_and(img,img,mask=mask_y)

    mask_v = cv2.inRange(hsv,lower_violet,upper_violet)
    output_v = cv2.bitwise_and(img,img,mask=mask_v)

    mask_rv = cv2.inRange(hsv,lower_redviolet,upper_redviolet)
    output_rv = cv2.bitwise_and(img,img,mask=mask_rv)
    dict = {'blue': output_b, 'light_violet': output_lv, 'red': output_r, 'green': output_g, 'yellow': output_y, 'violet': output_v, 'red_violet': output_rv}

    for i in dict:


        gray = cv2.cvtColor(dict[i], cv2.COLOR_BGR2GRAY)
        gau=cv2.GaussianBlur(gray,(5,5),0)


        bry2 = cv2.threshold(gau,45,255,cv2.THRESH_BINARY)[1]

        cnts1,hier1= cv2.findContours(bry2, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cv_contours = []
    
        for contour in cnts1:
            area = cv2.contourArea(contour)
            if area <= 120:
                cv_contours.append(contour)
            # x, y, w, h = cv2.boundingRect(contour)
            # img[y:y + h, x:x + w] = 255
            else:
                continue


        bry3=cv2.fillPoly(bry2,cv_contours,(255,255,255))
        cnts,hier= cv2.findContours(bry3, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


#cnt = cnts[0]
        for c in cnts:
            #if cv2.contourArea(c)>100:
            if c.size>120:

        
       
                M = cv2.moments(c)
#print( M )
                cX = int(M['m10']/M['m00'])
                cY = int(M['m01']/M['m00'])
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                cv2.circle(img,(cX,cY),7,(255,255,255),-1)
                cv2.putText(img,i,(cX-20, cY-20), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255),2)
                if cY<115:

                    k=str(s1)
                    cv2.putText(img,k,(cX+20, cY+20), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255),2)
                elif cY<336:
                    k=str(s2)
                    cv2.putText(img,k,(cX+20, cY+20), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255),2)
                elif cY<607:
                    k=str(s3)
                    cv2.putText(img,k,(cX+20, cY+20), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255),2)
                else:
                    k=str(s4)
                    cv2.putText(img,k,(cX+20, cY+20), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255),2)
                    
                cv2.imshow("video", img)

                if cv2.waitKey(2)>=27:
                    break
cam.release
cv2.destroyAllWindows()
exit
