#Setup Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers import Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

def model():
    learning_rate = 0.001
    ADAM = Adam(lr= learning_rate)
    #Build the model
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((72,25),(0,0))))

    model.add(Conv2D(24, (5,5), strides=(2,2), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Conv2D(36, (5,5), strides=(2,2), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Conv2D(48, (5,5), strides=(2,2), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer=ADAM)
    return model