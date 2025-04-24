import cv2 as cv
import matplotlib.pyplot as plt
import os


img = cv.imread("signatures/full_org/original_1_1.png")

def surf():
    root = os.getcwd()
    imgPath = os.path.join(root, "/SignatureFraudDetector/signatures/full_org/original_1_1.png")
    imgGray = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
    
    hessianThreshold = 3000
    surf = cv.xfeatures2d.SURF_create(hessianThreshold)
    keypoints = surf.detect(imgPath, None)
    imgGray = cv.drawKeypoints(imgGray, keypoints, imgGray, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.figure()
    plt.imshow(imgGray)
    plt.show()



if __name__ == "__main__":
        surf()