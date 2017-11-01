import numpy as np #matrix math
from sklearn.model_selection import train_test_split #to split out training and testing data
#keras is a high level wrapper on top of tensorflow (machine learning library)
#The Sequential container is a linear stack of layers
from keras.models import Sequential
#popular optimization strategy that uses gradient descent
from keras.optimizers import Adam
#to save our model periodically as checkpoints for loading later
from keras.callbacks import ModelCheckpoint
#what types of layers do we want our model to have?
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
#helper class to define input shape and generate training images given image paths & steering angles
from utils import INPUT_SHAPE, batch_generator, preprocess
#for reading files
import os
import csv
import cv2

def path_cleanup(path):

    filename = path.split('/')[-1]
    current_path = './data/IMG/' + filename
    return current_path

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

for i in range(len(lines)):
    for o in range(3):
        lines[i][o] = path_cleanup(lines[i][o])


model = Sequential()
model.add(Lambda(lambda x: (x/255)-0.5, input_shape=INPUT_SHAPE))
model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(64, 3, 3, border_mode='same', activation='elu'))
model.add(Conv2D(64, 3, 3, border_mode='same', activation='elu'))
model.add(Conv2D(64, 1, 1, border_mode='same', activation='elu'))
model.add(Conv2D(64, 3, 3, border_mode='same', activation='elu'))
model.add(Conv2D(64, 1, 1, border_mode='same', activation='elu'))
model.add(Conv2D(64, 3, 3, border_mode='same', activation='elu'))
model.add(Conv2D(64, 1, 1, border_mode='same', activation='elu'))
model.add(Conv2D(64, 3, 3, border_mode='same', activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(500, activation='elu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.2))
model.add(Dense(25, activation='elu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))
model.summary()


checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                             monitor='val_loss',
                             verbose=0,
                             save_best_only=True,
                             mode='auto')

model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))

model.fit_generator(batch_generator(lines, 128, True),
                    samples_per_epoch = 12800,
                    nb_epoch = 30,
                    max_q_size=1,
                    validation_data=batch_generator(lines, 64, False),
                    nb_val_samples=len(lines)*.2,
                    callbacks=[checkpoint],
                    verbose=1)
