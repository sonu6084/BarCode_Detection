import numpy as np 
import argparse
import cv2
import sys

#construct the argument parse and parse the arguments

ap = argparse.ArgumentParser()
# adding argument with flag "-i", og arg "--image" and help       
ap.add_argument("-i","--image",required = True,help = "path to the image file")
args = vars(ap.parse_args())

#loading file from the cmd 
# file = sys.argv[-1]

#load image 
image = cv2.imread(args['image'])
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# compute the gradient magnitude represntation of the image in 
# both x and y direction
gradX = cv2.Sobel(gray,ddepth = cv2.CV_32F,dx = 1,dy = 0,ksize = -1)
gradY = cv2.Sobel(gray,ddepth = cv2.CV_32F,dx = 0,dy = 1,ksize = -1)

#subtract the y-gradient from the x-gradient
gradient = cv2.subtract(gradX,gradY)
gradient = cv2.convertScaleAbs(gradient)

#blur and threshold the image
blurred = cv2.blur(gradient ,(9,9)) #kernel 9X9
_,thresh = cv2.threshold(blurred,210,255,cv2.THRESH_BINARY)
                                 # if 255 then set to 1 else 0

# constructing a closing kernel and apply it to the threshold 
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(21,7))
closed = cv2.morphologyEx(thresh , cv2.MORPH_CLOSE,kernel)

# perform a series of erosion and dilations
closed = cv2.erode(closed ,None, iterations = 4)
closed = cv2.dilate(closed ,None, iterations = 4)

# find the contour in the threshold image 
(cnts,_) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE )
c = sorted(cnts,key = cv2.contourArea, reverse=True)[0]

# compute the rotated bounding box of the largest contour
rect = cv2.minAreaRect(c)
box = np.int0(cv2.boxPoints(rect))

# draw a bounding box around the detected barcode 
cv2.drawContours(image,[box],-1,(0,255,0),3)

cv2.imshow("img",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
