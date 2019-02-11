import csv
import cv2 
import numpy as np
lines =[]
with open('/home/workspace/CarND-Behavioral-Cloning-P3/data/driving_log.csv') as csvfile: 
#with open('/root/Desktop/data/driving_log.csv') as csvfile:   
  reader = csv.reader(csvfile) 
  
  for line in reader:
    lines.append(line) 

images = [] 
measurements =[] 
# create adjusted steering measurements for the side camera images
correction = 2 # this is a parameter to tune

for line in lines[1:]:
  source_pathc = line[0] 
  source_pathl = line[1] 
  source_pathr = line[2]


  filenamec = source_pathc.split('/')[-1]
  filenamel = source_pathl.split('/')[-1]
  filenamer = source_pathr.split('/')[-1]
  current_pathc = 'data/IMG/' + filenamec 
  current_pathl = 'data/IMG/' + filenamel 
  current_pathr = 'data/IMG/' + filenamer 
  imagec = cv2.imread(current_pathc) 
  imagel = cv2.imread(current_pathl) 
  imager = cv2.imread(current_pathr) 
  
  #imagec.extend(imagel,imager)
  #images.append(imagec,imagel,imager)
  images.append(imagec)
  images.append(imagel)
  images.append(imager)
  measurementc = float(line[3])
  measurementl=measurementc+correction
  measurementr=measurementc-correction
  #measurementc.extend(measurementl,measurementr)
  measurements.append(measurementc)
  measurements.append(measurementl)
  measurements.append(measurementr)
augmented_images,augmented_measurements=[],[]
for image,measurement in zip(images,measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)
    
# Combine all the data
X_train = np.concatenate((images,augmented_images), axis=0)
y_train = np.concatenate((measurements,augmented_measurements), axis=0)    
#X_train=np.array(images)
#y_train=np.array(measurements)

from sklearn.utils import shuffle
# shuffle it 
X_train, y_train = shuffle(X_train, y_train)



from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation
from keras.layers.convolutional import Convolution2D, Cropping2D, ZeroPadding2D, MaxPooling2D

# Function to resize image to 64x64
def resize_image(image):
    import tensorflow as tf
    return tf.image.resize_images(image,[64,64])

model=Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(160,320,3)))
# Crop pixels from top and bottom of image
model.add(Cropping2D(cropping=((60, 20), (0, 0))))
# Resise data within the neural network
model.add(Lambda(resize_image))
# Normalize data
model.add(Lambda(lambda x: (x / 127.5 - 1.)))
# First convolution layer so the model can automatically figure out the best color space for the hypothesis
model.add(Convolution2D(3, 1, 1, border_mode='same', name='color_conv'))

# CNN model

model.add(Convolution2D(32, 3,3 ,border_mode='same', subsample=(2,2), name='conv1'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1), name='pool1'))

model.add(Convolution2D(64, 3,3 ,border_mode='same',subsample=(2,2), name='conv2'))
model.add(Activation('relu',name='relu2'))
model.add(MaxPooling2D(pool_size=(2,2), name='pool2'))

model.add(Convolution2D(128, 3,3,border_mode='same',subsample=(1,1), name='conv3'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size= (2,2), name='pool3'))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(128, name='dense1'))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(128))

model.add(Dense(1))



model.compile(loss='mse',optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=7)
model.save('model.h5')          
          
          
          
          