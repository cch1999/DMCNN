import os.path
import numpy as np
import cv2
import csv
#import keras
import matplotlib.pyplot as plt
import time

from tensorflow.python.keras.utils.data_utils import Sequence



import os.path
import math
import cv2

import numpy as np

import matplotlib.pyplot as plt
#from DMCNN.raw_precomputed_distance_matrix import raw_precomputed_distance_matrix

#----Directories----#

current_directory = os.path.dirname(os.path.abspath(__file__))
precomputed_path = os.path.join(current_directory, '../files/precomputed/')
precomputed_2_channels_path = os.path.join(current_directory, '../files/precomputed_2_channels/')

#----Distance Matrix Object-----#

class precomputed_distance_matrix(object):

	def __init__(self, pdb_id, dim, n_channels):
		self.dim = dim
		self.pdb_id = pdb_id
		self.n_channels = n_channels
		#Load precomputed matrix

		try:
			self.distance_matrix = np.load('/content/drive/My Drive/Colab Notebooks/Old_DMCNN/files/precomputed' + pdb_id.lower() + '.npy')

			self.distance_matrix = self.distance_matrix.reshape(*self.dim)

		except:
			"""
			mat = raw_precomputed_distance_matrix(pdb_id)
			mat.resize(self.dim[0])
			mat.set_hoizontal_to_one()
			mat.take_reciprocal()
			mat.set_hoizontal_to_zero()
			mat.half()
			mat.remove_noise(0.1)
			mat.normalise()
			self.distance_matrix = mat.distance_matrix
			self.save_precomputed_matrix()
			"""
			pass

	def save_precomputed_matrix(self):
		name = os.path.join(precomputed_path, self.pdb_id.lower() + '.npy')
		np.save(name, self.distance_matrix)

	def show(self):
		plt.imshow(self.distance_matrix, cmap="gray")
		plt.show()

#----Directories----#

current_directory = os.path.dirname(os.path.abspath(__file__))
precomputed_path = os.path.join(current_directory, 'files/precomputed/')

#----Data Generator----#

class DataGenerator(Sequence):

	def __init__(self, list_enzymes, labels, batch_size=32, dim=(900,900),
				 n_classes=2, n_channels= 1, directory_precomputed=precomputed_path, shuffle=True):
		'Initialization'
		self.dim = dim
		self.batch_size = batch_size
		self.labels = labels
		self.list_enzymes = list_enzymes
		self.n_classes = n_classes
		self.n_channels = n_channels
		self.shuffle = shuffle
		self.on_epoch_end()

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.list_enzymes))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, list_enzymes_temp):
		'Generates data containing batch_size samples' # X : (n_samples, *dim)
		# Initialization
		X = np.empty((self.batch_size, *self.dim[:2], self.n_channels))
		y = np.empty((self.batch_size), dtype=int)

		# Generate data
		for i, pdb_id in enumerate(list_enzymes_temp):
			if pdb_id == "":
				pdb_id = "3pew"
			else:
				pass

		# Augment and store sample, mat = matrix
			mat = precomputed_distance_matrix(pdb_id, self.dim, self.n_channels)

			if self.n_channels == 1:
				mat.distance_matrix = mat.distance_matrix.reshape(*self.dim)

			X[i,] = mat.distance_matrix

			# Store class
			y[i] = int(self.labels[pdb_id.upper()][1])

		return X, sparsify(y)

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.list_enzymes) / self.batch_size)) - 1

	def __getitem__(self, index):
		"Generate one batch of images"
		# Generate indexes of the batch
		indexes = self.indexes[(index-1)*self.batch_size:(index)*self.batch_size]

		# Generate pdb_ids for next batch
		list_enzymes_temp = [self.list_enzymes[k] for k in indexes]
		# Generate data
		X, y = self.__data_generation(list_enzymes_temp)

		return X, y



#----Misc Data Augmentation----#

def resize(distance_matrix, dim):
	return cv2.resize(distance_matrix, dim[:2])

def set_hoizontal_to_one(distance_matrix):
	for i in range(distance_matrix.shape[0]):
		distance_matrix[i,i] = 1
	return distance_matrix

def set_hoizontal_to_zero(distance_matrix):
	for i in range(distance_matrix.shape[0]):
		distance_matrix[i,i] = 0
	return distance_matrix

def take_reciprocal(distance_matrix):
	return distance_matrix**-1

def clear(distance_matrix, level):
	for i in range(distance_matrix.shape[0]):
		for j in range(distance_matrix.shape[1]):
			if distance_matrix[i,j] <= level:
				distance_matrix[i,j] = 0
			else:
				pass
	return distance_matrix

def half(distance_matrix):
	for i in range(distance_matrix.shape[0]):
		for j in range(distance_matrix.shape[1]):
			if j - i < 0:
				distance_matrix[i,j] = 0
			else:
				pass
	return distance_matrix

def read_dict(path):
    'Reads Python dictionary stored in a csv file'
    dictionary = {}
    for key, val in csv.reader(open(path)):
        dictionary[key] = val
    return dictionary

def read_list(path):
	"Reads list stored in txt file"
	list_enzymes = open(path,'r')
	list_enzymes = list_enzymes.read()
	return list(list_enzymes.split(", "))

def sparsify(y):
    'Returns labels in binary NumPy array'
    n_classes = 6
    return np.array([[1 if y[i] == j+1 else 0 for j in range(n_classes)]
                      for i in range(y.shape[0])])
