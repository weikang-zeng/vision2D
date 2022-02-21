  # -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

"""
VISION 2D:
Training code
"""
import cv2
"""
click callback
"""
import numpy as np
"""
for shape detector
"""

import argparse
import imutils
"""
Reading a single image
"""
# Read and display BGR image
img = cv2.imread ( './paperEval.png' )


#记录第一行的y在105～115之间，x随意变化
#第二行y336
#第三行y 607


# if (not bgr.data):
#     print ('Impossible to read image !')

#a = []
#b = []

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--input", required=True,
#     help="./paperAppr.png")
# args = vars(ap.parse_args())
 
# # 进行灰度化得到二值图
# image = cv2.imread(args["input"])
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

# cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
#     cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if imutils.is_cv2() else cnts[1]


# for c in cnts:
#     # 获取中心点
#     M = cv2.moments(c)
#     cX = int(M["m10"] / M["m00"])
#     cY = int(M["m01"] / M["m00"])
 
#     # 画出轮廓和中点
#     cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
#     cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
#     cv2.putText(image, "center", (cX - 20, cY - 20),
#         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
 
#     #显示图像
#     cv2.imshow("Image", image)
#     cv2.waitKey(0)
#     颜色边界检测
#     https://blog.csdn.net/weixin_42216109/article/details/89520423
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray_image,127,255,0)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_violet=cv2.inRange(hsv,(152,220,136),(162,248,255))
img_specifiedColor=cv2.bitwise_and(img,img,mask=img_violet)
img_blue=cv2.inRange(hsv,(112,196,190),(115,223,255))
img_specifiedColor1=cv2.bitwise_and(img,img,mask=img_blue)

#显示颜色字
M = cv2.moments(thresh)
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])
#cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
cv2.putText(img_specifiedColor, "violet", (cX - -125, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#cv2.imshow("Image", img)


a=0;
b=0;
c=0;
d=0;
e=0;
f=0;



def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        global a,b,c,d,e,f
        #a.append(x)
        #b.append(y)
        cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)
        #print(x,y)
        #(b,g,r)=img[x,y]
        #print("red: %d,green:%d,blue:%d "%(r,g,b))
        print("HSV:", hsv[y, x])
        (h,s,v)=hsv[y,x]
        #print("H: %d,S:%d,V:%d "%(h,s,v))
        print(h)
        if h>a:
            a=h
        elif s>b:
            b=s
        elif v>c:
            c=v
        print(a,b,c)
        if h<a:
            d=h
        elif s<b:
            e=s
        elif v<c:
            f=v
        print(d,e,f)
        #b=a;
        #h=np.append(a,b)
        #print(h)
        
        

cv2.namedWindow("image")
cv2.setMouseCallback("image",on_EVENT_LBUTTONDOWN) 
   
while(1):
    cv2.imshow("image", img)
    #cv2.imshow("image_vio", img_specifiedColor)
    #cv2.imshow("image_blue", img_specifiedColor1)
    if cv2.waitKey(0)&0xFF==27:
        break
    #print(a[0],b[0])
cv2.destroyAllWindows()
exit
# print(bgr.shape)
# print(bgr[10,10]







# # Convert BGR into HSV and display
# hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
# cv2.imshow("HSV image", hsv)
# cv2.waitKey(0)
# # Extract hue and display
# h,s,v = cv2.split(hsv)
# cv2.imshow("Hue image", h)
# cv2.waitKey(0)
# # Convert BGR into GRAYSCALE and display
# gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Gray image", gray)
# cv2.waitKey(0)
# # Display grayscale image with a color map
# clr = cv2.applyColorMap( gray, cv2.COLORMAP_JET )
# cv2.imshow("Color map image", clr)
# cv2.waitKey(0)
# # Global thresholding of GRAYSCALE and display
# rv, bry = cv2.threshold(gray, 0,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# cv2.imshow("Binary image", bry)
# cv2.waitKey(0)
# # Local thresholding of GRAYSCALE and display
# bry2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
# cv2.THRESH_BINARY, 11, 2)
# cv2.imshow("Binary image 2", bry2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

