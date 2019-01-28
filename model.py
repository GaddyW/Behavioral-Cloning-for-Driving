#Todos
    #read up on various models
    #use right and left cameras - maybe
    #bird's eye view?
      
    

#DONE
    #use both data sets
    #batch size 128
    #use generator
    #crop the image
    #flip images and invert direction for augmentation
    #BGR to RGB
    #reshape down
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
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt
from tqdm import tqdm

#import data
def inputdata(logfile, path, samples):
    counter = 0
    with open(logfile) as csvfile:
        reader = csv.reader(csvfile)
        for line in tqdm(reader):
            if (line[0] == 'center'):
                continue
            line[0] = path  + line[0]
            #print(path)
            #print(line[0])
            lines.append(line)
            #counter += 1
            #if counter > 140:
                 #return lines
    print(len(lines))
    return lines       

lines = []
#lines = inputdata('./SmallData/driving_log.csv', './SmallData/', lines)

print('Their data')
lines = inputdata('./data/driving_log.csv', './data/', lines)

print('Recovery data')
lines = inputdata('./recovery/driving_log.csv', '', lines)
#lines = inputdata('./recovery/driving_log.csv', '', lines)

print('My data')
lines = inputdata('/home/opt/MyTraining/driving_log.csv', '/home', lines)

print('Track 2')
lines = inputdata('/home/track2/driving_log.csv', '', lines)

                
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
print('Training set size: %s' %len(train_samples))
print('Validation set size: %s' %len(validation_samples))


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                source_path = batch_sample[0]
                #filename = source_path.split('/')[-1]
                
                if (("2019_01_22_08_55_15" in source_path)|("2019_01_22_08_55_16" in source_path)|("2019_01_22_08_55_17" in source_path)|(not os.path.isfile(source_path))):
                    continue
                #current_path = filename
                #print(source_path)
                bgr_image = cv2.imread(source_path)
                b,g,r = cv2.split(bgr_image)       # get b,g,r
                image = cv2.merge([r,g,b])     # switch it to rgb
                #image = cv2.resize(image,(160, 80), interpolation = cv2.INTER_CUBIC)
                angle = float(batch_sample[3])
                
                #augment images
                images.append(image)
                angles.append(angle)
                images.append(cv2.flip(image,1))
                angles.append(angle*-1.0)
                

            # put data in form that Keras can handle
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
BS = 128
train_generator = generator(train_samples, batch_size=BS)
validation_generator = generator(validation_samples, batch_size=BS)            
            

#build network architecture
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D


model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/255 - 0.5))
model.add(Conv2D(filters = 24, kernel_size = 5, strides=(2, 2), activation="relu"))
model.add(Conv2D(filters = 36, kernel_size = 5, strides=(2, 2), activation="relu"))
model.add(Conv2D(filters = 48, kernel_size = 5, strides=(2, 2), activation="relu"))
model.add(Conv2D(filters = 64, kernel_size = 3, activation="relu"))
model.add(Conv2D(filters = 64, kernel_size = 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


#train network
from keras.models import Model


model.compile(loss='mse', optimizer = 'adam')
#history_object = model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, epochs = 5, verbose = 1)
history_object = model.fit_generator(train_generator, steps_per_epoch = len(train_samples) // BS, 
                                     validation_data = validation_generator, validation_steps = len(validation_samples) // BS, 
                                     epochs=5, verbose=1)


#save model
model.save('model.h5')


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