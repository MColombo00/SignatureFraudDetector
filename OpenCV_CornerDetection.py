import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
 
img_orig = cv.imread("./original_10_1.png")  # Read image data
img_forg = cv.imread("./forgeries_10_1.png")


def goodCornerDetection(img):
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    maxCorners = 200
    quality = 0.05  # Controls amount of dots; 0.05 was a good compromise btwn amount and accuracy
    minDistance = 20
    
    corners = cv.goodFeaturesToTrack(imgGray, maxCorners, quality, minDistance)
    
    for corner in corners:
        x = int(corner[0][0])
        y = int(corner[0][1])
        cv.circle(imgRGB, (x,y),3,(0,0,255),-1) # 3rd value is size of dots
    return imgRGB


# Output new created images with the new lines superimposed
cv.imwrite('./LINES_original.png', goodCornerDetection(img_orig))
cv.imwrite('./LINES_forg.png', goodCornerDetection(img_forg))

print("Complete.")
