import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
 
img_orig = cv.imread(r"signatures/full_org/original_10_1.png")  # Read image data
img_forg = cv.imread(r"signatures/full_forg/forgeries_10_1.png")


def goodCornerDetection(img, name, dirIn, dirOutDat, dirOutImg, pointFile):
    data_list = []
    print(dirOutImg)
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
    
    # print(data_list)

    with open(f"{dirOutDat}/{pointFile}", "a") as outfile:
        for x in data_list:
            outfile.write(f"{str(x[0])},{str(x[1])}|")
        outfile.write("\n")
    
    # cv.imwrite(f"{dirOutImg}/{name}",imgRGB)
    


def siftTest():     #using SIFT algorithm for feature detection
    img = img_orig
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    
    sift = cv.SIFT_create()
    kp = sift.detect(gray,None)
    
    print(kp[0])
    img=cv.drawKeypoints(gray,kp,img)
    
    # cv.imwrite('sift_keypoints.jpg',img)

# Output new created images with the new lines superimposed
# cv.imwrite(r'./LINES_original.png', goodCornerDetection(img_orig))
# cv.imwrite(r'./LINES_forg.png', goodCornerDetection(img_forg))

def main():
    inputPathName = ["./signatures/full_org","./signatures/full_forg"]
    outputCornerPathName = [("./CornerDetect/data","./CornerDetect/full_org"),
                         ("./CornerDetect/data","./CornerDetect/full_forg")]
    
    
    # scan = os.listdir(inputPathName[0])
    # print(os.getcwd())
    for x in range(len(inputPathName)):
        for dirpath, dirnames, filenameList in os.walk(inputPathName[x],topdown=True):
            for filename in filenameList:
                if filename.endswith(".png"):
                    img = cv.imread(f"{inputPathName[x]}/{filename}")
                    if "full_org" in inputPathName[x]:
                        pointFile = "org_data_cords.csv"
                    else:
                        pointFile = "forg_data_cords.csv"
                    
                    goodCornerDetection(img,filename,inputPathName[x],outputCornerPathName[x][0],outputCornerPathName[x][1], pointFile)
            


    # path = "./signatures/full_org"
    
    # files = []
    # count = 0
    # for pic in scan:
    #     count += 1
    # print(count)
    # goodCornerDetection(img_orig)
    # siftTest()
    
    
    pass

main()
print("Complete.")