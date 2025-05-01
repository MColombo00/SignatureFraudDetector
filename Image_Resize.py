import numpy as np
import os
import sklearn
import skimage
from matplotlib import pyplot as plt
x = []
y = [0, 1] #0 for original, 1 for forgery

def generateData(file, label):
    img = skimage.io.imread(f"{file}", as_gray=True)

    resizeImg = skimage.img_as_ubyte(skimage.transform.resize(img, (256, 256)))
    flatImg = resizeImg.flatten()
    normalizedImg = flatImg / 255.0

    x.append(normalizedImg)
    y.append(label)



inputPathName = ["./signatures/full_org", "./signatures/full_forg"]
outputPathName = ["./TestDataSkimage/full_org", "./TestDataSkimage/full_forg"]
for x in range(len(inputPathName)):
    for dirpath, dirnames, filenameList in os.walk(inputPathName[x], topdown=True):
        for filename in filenameList:
            if filename.endswith(".png"):
                generateData(filename, x)



# print(x[0])
# print(flatImg)
# skimage.io.imsave("./test.png", resizeImg)
# plt.imshow(resizeImg, cmap='grey')
# plt.show()
