import numpy as np
import os
import skimage
from matplotlib import pyplot as plt
count = 0
x = []
y = []  # 0 for original, 1 for forgery


def generateData(path , file, label):
    img = skimage.io.imread(os.path.join(path, file), as_gray=True)

    resizeImg = skimage.img_as_ubyte(skimage.transform.resize(img, (256, 256)))

    if (label == 0):
        skimage.io.imsave(f"./TestDataSkimage/full_org/{file}_Processed.png", resizeImg)      
    elif (label == 1):
        skimage.io.imsave(f"./TestDataSkimage/full_forg/{file}_Processed.png", resizeImg)

    flatImg = resizeImg.flatten()
    normalizedImg = flatImg / 255.0

    return normalizedImg


inputPathName = ["./signatures/full_org", "./signatures/full_forg"]
outputPathName = ["./TestDataSkimage/full_org", "./TestDataSkimage/full_forg"]
for i in range(len(inputPathName)):
    for dirpath, dirnames, filenameList in os.walk(inputPathName[i], topdown=True):
        for filename in filenameList:
            if filename.endswith(".png"):
                img = generateData(dirpath, filename, i)
                x.append(img)
                y.append(i)


np.save("./TestDataSkimage/data/X.npy", x, allow_pickle=True)
np.save("./TestDataSkimage/data/Y.npy", y, allow_pickle=True)
