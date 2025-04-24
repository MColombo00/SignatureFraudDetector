import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
 
img_orig = cv.imread(r"signatures/full_org/original_10_1.png")  # Read image data
img_forg = cv.imread(r"signatures/full_forg/forgeries_10_1.png")


def goodCornerDetection(img):
    data_list = []
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    maxCorners = 200
    quality = 0.05  # Controls amount of dots; 0.05 was a good compromise btwn amount and accuracy
    minDistance = 20
    
    corners = cv.goodFeaturesToTrack(imgGray, maxCorners, quality, minDistance)
    
    for corner in corners:
        x = int(corner[0][0])
        y = int(corner[0][1])
        data_list.append((x,y))
        cv.circle(imgRGB, (x,y),3,(0,0,255),-1) # 3rd value is size of dots
    
    print(data_list)
    with open(f"./postSignature/test.dat", "w") as outfile:
        for x in data_list:
            outfile.write(f"{str(x[0])},{str(x[1])}\n")

    return imgRGB


# Output new created images with the new lines superimposed
# cv.imwrite(r'./LINES_original.png', goodCornerDetection(img_orig))
# cv.imwrite(r'./LINES_forg.png', goodCornerDetection(img_forg))

def main():
    path = "./signatures/full_forg"
    scan = os.scandir(path)
    files = []
    count = 0
    for pic in scan:
        count += 1
    print(count)


    
    
    pass

main()
print("Complete.")