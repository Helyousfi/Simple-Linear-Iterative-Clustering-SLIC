# import libraries
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap, BoundaryNorm
from PIL import Image

image = Image.open("dog.jpg")
image = np.array(image)
image = image[..., 0]
print(image.shape)

def distance_pixels(pixel1, pixel2, m, S):
    d_lab = np.sqrt(
        (pixel1[2] - pixel2[2]) ** 2
    )

    d_xy = np.sqrt(
        (pixel1[0] - pixel2[0]) ** 2
        + (pixel1[1] - pixel2[1]) ** 2
    )

    D_s = d_lab + (m/S) * d_xy
    return D_s

height, width = image.shape

pixels_5d = np.zeros((height, width, 3))

for i in range(height):
    for j in range(width):
        pixels_5d[i, j, 0] = i
        pixels_5d[i, j, 1] = j
        pixels_5d[i, j, 2] = image[i, j] # if RGB add another channel


# Initializing cluster centers
S = 8
Ck = np.zeros((S+1,S+1,3))

index_i = 0
for i in range(0, height, int(height//S)):
    index_j = 0
    for j in range(0, width, int(width//S)):
        Ck[index_i, index_j, 0] = i
        Ck[index_i, index_j, 1] = j
        Ck[index_i, index_j, 2] = image[i, j]
        index_j += 1
    index_i += 1

super_pixel_image = np.zeros_like(image)
for i in range(height):
    for j in range(width):
        dist = np.zeros((S,S))
        for m in range(S):
            for n in range(S):
                dist[m,n] = distance_pixels(Ck[m,n], pixels_5d[i,j], 20, S)
        arg_min = np.argmin(dist)
        row, column = arg_min // S, arg_min % S
        super_pixel_image[i,j] = Ck[row, column, 2]

    
plt.imshow(super_pixel_image, cmap="gray")
plt.show()