import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import keras
import time
import os.path
from keras import backend as K
from data_generator import read_dict, read_list, DataGenerator
from DMCNN.precomputed_distance_matrix import precomputed_distance_matrix
from DMCNN.raw_precomputed_distance_matrix import raw_precomputed_distance_matrix
import matplotlib.pyplot as plt

NAME = "models/" + "DMCNN_1567754375"

#--------Parameters--------#

IMG_SIZE = 100

#----Dataset----#

training_list = read_list('datasets/training.txt')
validation_list = read_list('datasets/validation.txt')

labels = read_dict("datasets/dataset_all.csv")


#------------Model------------#

model = tf.keras.models.load_model(NAME)

test_enzymes = validation_list

total = list((0,0,0,0,0,0))
correct = list((0,0,0,0,0,0))

def sparsify(y):
    'Returns labels in binary NumPy array'
    n_classes = 6
    return np.array([[1 if y[i] == j+1 else 0 for j in range(n_classes)]
                      for i in range(y.shape[0])])


for pdb_id in test_enzymes:

    if pdb_id == "":
        pdb_id = "3pew"
    else:
        pass
    # Augment and store sample, mat = matrix
    mat = precomputed_distance_matrix(pdb_id, (100,100), n_channels=1)

    x = mat.distance_matrix
    x = x.reshape(1,100,100,1)

    prediction = model.predict(x)

    predicted_class = np.argmax(prediction) + 1

    labeled_class = int(labels[pdb_id.upper()][1])

    total[labeled_class-1] = total[labeled_class-1] + 1

    if predicted_class == labeled_class:
        print("Correct!")
        correct[labeled_class-1] = correct[labeled_class-1] + 1
    else:
        print("Incorrect")
y_label = [0 for i in range(7)]
y_percentage = [0 for i in range(7)]

for i in range(6):
    y_label[i] = str(correct[i]) + "/" + str(total[i])
    y_percentage[i] = correct[i] / total[i] * 100
    print(y_label[i])
    print(y_percentage[i])

total_correct = sum(correct)
y_percentage[6] = total_correct / len(test_enzymes) * 100
y_label[6] = str(total_correct) + "/" + str(len(test_enzymes))

classes = ("EC1", "EC2", "EC3", "EC4", "EC5", "EC6", "Total")


ax = plt.subplot()

ax.bar(classes, y_percentage, align="center", alpha=0.5)

ax.set_ylabel("Percentage accuracy")
ax.set_title("Accuracy per enzyme class")
#ax.set_xticklabels(("EC1", "EC2", "EC3", "EC4", "EC5", "EC6"))
rects = ax.patches

# Make some labels.
labels = ["{percentage}%\n{label}".format(percentage=int(y_percentage[i]), label=y_label[i]) for i in range(len(classes))]

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height - 12, label,
            ha='center', va='bottom')
plt.show()
