import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

def gen_labels():
    return {
        0: 'cardboard',
        1: 'glass',
        2: 'metal',
        3: 'paper',
        4: 'plastic',
        5: 'trash',
    }

def preprocess(image):
    image = np.array(image.resize((300, 300), Image.LANCZOS))
    image = np.array(image, dtype='uint8')
    image = np.array(image)/255.0

    return image

def model_arc():
    model = Sequential()

    # Convolution blocks
    model.add(Conv2D(32, kernel_size=(3,3), padding='same', input_shape=(300,300,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(32, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    # Classification layers
    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))

    model.add(Dropout(0.2))
    model.add(Dense(6, activation='softmax'))

    # Enable OneDNN optimizations
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    

    return model
