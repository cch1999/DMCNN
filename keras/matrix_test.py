import os.path
import math

import numpy as np
import cv2
from DMCNN.raw_precomputed_distance_matrix import raw_precomputed_distance_matrix
import matplotlib.pyplot as plt

def show(mat):
    plt.imshow(mat, cmap="gray")
    plt.show()

mat = raw_precomputed_distance_matrix("1o6y")

mat.resize(100)

mat.set_hoizontal_to_one()

mat.take_reciprocal()
mat.set_hoizontal_to_zero()



mat.half()




mat.remove_noise(0.05)
mat.normalise()
mat.show()
print(mat.distance_matrix.shape)
matrix = np.empty((100,100,3))
for i in range(2):
	matrix[:,:,i] = mat.distance_matrix

print(matrix.shape)


matrix[:,:,2] = 0
for i in range(100):
	for j in range(100):
		if abs(i-j) <= 5:
			matrix[i,j,0] = 0

for i in range(100):
	for j in range(100):
		if abs(i-j) > 5:
			matrix[i,j,1] = 0

show(matrix)
show(matrix[:,:,0])
show(matrix[:,:,1])
