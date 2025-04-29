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
        test = [452,186430,194130,173516,188101,164629,183162,166258,192330,181373,19372,171450,165518,157604,184104,205141,202381,16349,154487,17984,208645,216418,169656,242258,1632,115655,144166,129405,19936,206655,323300,210208,171186,157565,190138,141232,199606,162655,444163,188656,3847,224655,175279,17041,129655,38356,107237,175203,207209,150656,64357,86656,346655,9202,97561,154656,96597,385347,141227,441368,213656,464656,404102,185584,304584,339541,189173,20722,151655,198475,196347,19616,180656,265231,146620,375641,3053,13918,311610,22761,201298,186316,19736,2903,2579,202587,267363,128594,245635,333201,130656,286280,200505,254655,120173,286395,377585,36941,181]
        