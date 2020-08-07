import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, LeakyReLU
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
#import keras
import time
import os.path
from tensorflow.keras import backend as K
from data_generator import read_dict, read_list, DataGenerator
from DMCNN.tools import get_class_weights

NAME = "DMCNN_{}".format(int(time.time()))

#----TensorBoard----#

tensorboard = TensorBoard(log_dir="logs\{}".format(NAME))

#--------Parameters--------#

IMG_SIZE = 100

class_weights_mode = "balanced"

#----Model Hyperparameters----#

two_channel = False
batch_size = 128
max_epochs = 20
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

print(len(training_list))
print(len(validation_list))

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

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(5,5), input_shape = (IMG_SIZE,IMG_SIZE,n_channels)))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(rate=0.2))

model.add(Conv2D(filters=16, kernel_size=(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(rate=0.2))

model.add(Conv2D(filters=16, kernel_size=(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(rate=0.2))

model.add(Flatten())

model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(rate=0.4))

model.add(Dense(32))
model.add(Activation("relu"))
model.add(Dropout(rate=0.4))

model.add(Dense(n_classes))
model.add(Activation("softmax"))


# Track accuracy and loss in real-time
#history = MetricsHistory(saving_path=NAME + '.csv')

# Checkpoints
#checkpoints = ModelCheckpoint('checkpoints/' + NAME + '_{epoch:02d}' + '.hd5f',
                        #      save_weights_only=True,
                        #      period=period_checkpoint)

model.compile(loss="categorical_crossentropy",
            optimizer=Adam(lr=0.0004),
            metrics=["accuracy"])

model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=max_epochs,
                    max_queue_size=16,
                    class_weight=class_weights,
                    callbacks=[tensorboard])
                    #use_multiprocessing=True,
                    #workers=2)

model.save("models/" + NAME)
print("\nSaved " + NAME)

K.clear_session()
