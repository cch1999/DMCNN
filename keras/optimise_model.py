import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, LeakyReLU
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
import keras
import time
import os.path
from tensorflow.keras import backend as K
from data_generator import read_dict, read_list, DataGenerator
from DMCNN.tools import get_class_weights

#--------Parameters--------#

IMG_SIZE = 100

class_weights_mode = "balanced"

#----Model Hyperparameters----#

two_channel = False
batch_size = 128
max_epochs = 15
n_classes = 6

if two_channel == True:
    n_channels = 2
else:
    n_channels = 1

#----Dataset----#

reduced = False

if reduced == True:
    training_list = read_list('datasets/training_2,3.txt')
    validation_list = read_list('datasets/validation_2,3.txt')
else:
    training_list = read_list('datasets/training.txt')
    validation_list = read_list('datasets/validation.txt')

labels = read_dict("datasets/dataset_all.csv")

class_weights = get_class_weights(labels, training_list, mode=class_weights_mode)

#--------Generators--------#

training_generator = DataGenerator(training_list, labels,
                                        n_classes=n_classes,
                                        n_channels=n_channels,
                                        batch_size=batch_size,
                                        dim=(IMG_SIZE, IMG_SIZE,1))

validation_generator = DataGenerator(validation_list, labels,
                                        n_classes=n_classes,
                                        n_channels=n_channels,
                                        batch_size=batch_size,
                                        dim=(IMG_SIZE, IMG_SIZE,1))

#------------Model------------#

filters1 = [64, 100, 128]
filters2 = [32, 64]
dense = [64, 128, 256]
dropout = [0.2, 0.4, 0.6]


for filter1 in filters1:
    for filter2 in filters2:
        for dense_size in dense:
            for dropout_value in dropout:

                NAME = "DMCNN_filter_1_{filter1}_filter_2_{filter2}_dense_{dense_size}_dropout_{dropout_value}".format(filter1 = filter1,
                                                                            filter2 = filter2,
                                                                            dense_size=dense_size,
                                                                            dropout_value=dropout_value)

                print(NAME)

                tensorboard = TensorBoard(log_dir="logs\{}".format(NAME))

                model = Sequential()
                model.add(Conv2D(filters=filter1, kernel_size=(5,5), input_shape = (IMG_SIZE,IMG_SIZE,n_channels)))
                model.add(LeakyReLU(alpha=0.1))
                model.add(Dropout(rate=0.2))


                model.add(Conv2D(filters=filter2, kernel_size=(3,3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Dropout(rate=0.2))

                model.add(Flatten())

                model.add(Dense(dense_size))
                model.add(Activation("relu"))
                model.add(Dropout(rate=dropout_value))

                model.add(Dense(n_classes))
                model.add(Activation("softmax"))

                model.compile(loss="categorical_crossentropy",
                            optimizer="adam",
                            metrics=["accuracy"])
                print(NAME)
                model.fit_generator(generator=training_generator,
                                    validation_data=validation_generator,
                                    epochs=max_epochs,
                                    max_queue_size=16,
                                    class_weight=class_weights,
                                    callbacks=[tensorboard])
                                    #use_multiprocessing=True,
                                    #workers=2)
                print(NAME)
                model.save("models/" + NAME)

                K.clear_session()
