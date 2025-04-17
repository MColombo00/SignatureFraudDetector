import cv2 as cv
import numpy as np
 
img = cv.imread("./original_10_1.png")  # Read image data
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)  # Make image grayscale (no change to the signatures images)
edges = cv.Canny(gray,50,150,apertureSize = 3)  # The edge-detection function
print(edges)

lines = cv.HoughLines(edges,1,np.pi/180,200)  # The line-detection functiton building off of the detected edges
print(lines)

# Commented out bc it gives an error for the signature images (though a basic scribbles image I made in Paint works just fine)
# The idea was to take the line data and use it to draw the lines onto the image:
'''for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
 
    cv.line(img,(x1,y1),(x2,y2),(255,0,0),2)  # Actually draws the line
 '''
cv.imwrite('./LINES_original.png',gray)  # Output new created image with the new lines superimposed



