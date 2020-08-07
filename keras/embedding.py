import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from DMCNN.precomputed_distance_matrix import precomputed_distance_matrix
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data
from data_generator import read_dict, read_list

# import the data and split into X and Y



embed_count = 160
reduced = False

training_list = read_list('datasets/training.txt')
validation_list = read_list('datasets/validation.txt')

labels = read_dict("datasets/dataset_all.csv")

logdir = r'C:\Users\Charlie\OneDrive - Imperial College London\Python\DMCNN\emb_log'  # you will need to change this!!!

x_test = np.empty((embed_count, 100, 100))
y_test = np.empty((embed_count))

for i in range(embed_count):
    img = precomputed_distance_matrix(training_list[i], (100,100))
    x_test[i] = img.distance_matrix
    y_test[i] = labels[training_list[i].upper()][1]

# setup the write and embedding tensor

summary_writer = tf.summary.FileWriter(logdir)

embedding_var = tf.Variable(x_test, name='DMCNN_embedding')

config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name

embedding.metadata_path = os.path.join(logdir, 'metadata.tsv')

projector.visualize_embeddings(summary_writer, config)

# run the sesion to create the model check point

with tf.Session() as sesh:
    sesh.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver()
    saver.save(sesh, os.path.join(logdir, 'model.ckpt'))

# create the sprite image and the metadata file

rows = 100
cols = 100

label = ['1', '2', '3', '4', '5', '6']



index = 0
labels = []
for i in range(embed_count):
    labels.append(int(y_test[index]))

    index += 1

with open(embedding.metadata_path, 'w') as meta:
    meta.write('Index\tLabel\n')
    for index, label in enumerate(labels):
        meta.write('{}\t{}\n'.format(index, label))
