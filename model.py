#Todos
    #read up on various models
    #use right and left cameras - maybe
    #bird's eye view?
    #use generator

#DONE
    #crop the image
    #flip images and invert direction for augmentation
    #skip the time I drove off the track
        ####center_2019_01_22_08_55_15_013.jpg
        ####center_2019_01_22_08_55_17_260.jpg
        ####2019_01_22_08_55_15, 2019_01_22_08_55_16, 2019_01_22_08_55_17

import csv
import os
import cv2
import numpy as np
import sklearn
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from tqdm import tqdm

#import data
lines = []
with open('./SmallData/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in tqdm(reader):
        lines.append(line)

images = []
angles = []
counter = 0

for line in tqdm(lines):
    source_path = line[0]
    filename = source_path.split('/')[-1]
    if ((source_path == 'center')|("2019_01_22_08_55_15" in filename)|("2019_01_22_08_55_16" in filename)|("2019_01_22_08_55_17" in filename)):
        continue

    current_path = './SmallData/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    angle = float(line[3])
    angles.append(angle)
    counter += 1
    if counter > 140: 
        break


# compile and train the model using the generator function
#train_generator = generator(train_samples, batch_size=32)
#validation_generator = generator(validation_samples, batch_size=32)    

#preprocess data in batches in a generator
#camera calibration?
#bird's eye view?

#augment data 
augmented_images, augmented_angles = [], []
for image, angle in zip(images, angles):
    augmented_images.append(image)
    augmented_angles.append(angle)
    augmented_images.append(cv2.flip(image,1))
    augmented_angles.append(angle*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_angles)
cv2.imwrite('test.jpg', augmented_images[150])


#build network architecture
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D


model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/255 - 0.5))
model.add(Flatten())
model.add(Dense(1))

#train network
from keras.models import Model


model.compile(loss='mse', optimizer = 'adam')
history_object = model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, epochs = 5, verbose = 1)
#history_object = model.fit_generator(train_generator, samples_per_epoch =
#    len(train_samples), validation_data = 
#    validation_generator,
#    nb_val_samples = len(validation_samples), 
#    nb_epoch=5, verbose=1)

### print the keys contained in the history object
#print('Loss %s' %(history_object.history['loss']))
#print('Val Loss %s' %(history_object.history['val_loss']))

### plot the training and validation loss for each epoch
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show(block=True)

#save model
model.save('model.h5')