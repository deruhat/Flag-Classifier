from __future__ import print_function
import keras
from keras.models import load_model
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard

from time import time

# Data generators
train = ImageDataGenerator(rotation_range = 45.0,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    horizontal_flip = True,
    shear_range = 30.,
    zoom_range = 0.3)

test = ImageDataGenerator(rotation_range = 45.0,
    zoom_range = 0.3)

# Target directories
trainGenerator = train.flow_from_directory('data/train', target_size = (20, 20),
    color_mode = 'rgb', seed = 123)

testGenerator = test.flow_from_directory('data/test', target_size = (20, 20),
    color_mode = 'rgb', seed = 777)

validationGenerator = test.flow_from_directory('data/validation', target_size = (20, 20),
    color_mode = 'rgb', seed = 987)

# Neural network
model = Sequential()
model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding = 'same', input_shape = (20, 20, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters = 128, kernel_size = (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dropout(0.1))

model.add(Dense(17)) 
model.add(Activation('softmax'))

# Print a summary of the network
print(model.summary())

# Set optimizer and loss function
opt = keras.optimizers.SGD(lr = 0.0001, momentum = 0.01)
optt = keras.optimizers.RMSprop(lr = 0.0001)
model.compile(loss='categorical_crossentropy',
            optimizer=optt,
            metrics=['categorical_accuracy'])

# tensorboard
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

# Fit the model!
model.fit_generator(trainGenerator,
            steps_per_epoch = 100,
            epochs = 120,
            validation_data = validationGenerator,
            validation_steps = 40,
            callbacks=[tensorboard])

result = model.evaluate_generator(testGenerator)
print(result)

model.save('saved_model.h5')