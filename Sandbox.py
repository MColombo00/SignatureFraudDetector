import skimage
import sklearn
import os
from matplotlib import pyplot as plt

originalImages = os.listdir("signatures/full_org/")

img = skimage.io.imread("signatures/full_org/original_1_1.png", as_gray=True)
img = skimage.feature.canny(img)

height, width = img.shape
cell_height = height // 6
cell_width = width // 6

corner_coords = []

for row in range(6):
    for col in range(6):
        y_start = row * cell_height
        y_end = (row + 1) * cell_height
        x_start = col * cell_width
        x_end = (col + 1) * cell_width

        cell = img[y_start:y_end, x_start:x_end]
        coords = skimage.feature.corner_peaks(
            skimage.feature.corner_shi_tomasi(cell), min_distance=10)
        for (cornerY, cornerX) in coords:
            corner_coords.append((cornerY + y_start, cornerX + x_start))

skimage.io.imshow(img)
print(height)
print(corner_coords)
for (cornerY, cornerX) in corner_coords:
    plt.plot(cornerX, cornerY, "og", markersize=5)
plt.show()
print(img)
