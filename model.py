import os
import numpy as np
from PIL import Image
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


gestures = {'A': 'A', 'E': 'E', 'H': 'H', 'L': 'L', 'O': 'O'}
gestures_map = {'A': 0, 'E': 1, 'H': 2, 'L': 3, 'O': 4}
image_path = 'D:/XLA&TGMT/recognize_sign/Dataset'
models_path = 'models/saved_model2.keras'  
image_size = 224


def import_image(path):
    img = Image.open(path)
    img = img.resize((image_size, image_size))
    img = np.array(img)
    return img

def import_data(X_data, y_data):
    X_data = np.array(X_data, dtype='float32')
    if X_data.shape[-1] != 3:  
        raise ValueError("Input images RGB")
    X_data /= 255  
    y_data = np.array(y_data)
    y_data = to_categorical(y_data)  
    return X_data, y_data

def file_data(image_path):
    X_data = []
    y_data = []
    for directory, subdirectories, files in os.walk(image_path):
        gesture_name = os.path.basename(directory)  
        if gesture_name in gestures_map:  
            for file in files:
                if not file.startswith('.'):  
                    path = os.path.join(directory, file)
                    y_data.append(gestures_map[gesture_name])
                    X_data.append(import_image(path))
        else:
            print(f"Warning: {directory}")  

    if len(X_data) == 0 or len(y_data) == 0:
        raise ValueError("Empty dataset found.")
    
    X_data, y_data = import_data(X_data, y_data)
    return X_data, y_data

X_data, y_data = file_data(image_path)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=12, stratify=y_data)

model_checkpoint = ModelCheckpoint(filepath=models_path, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, restore_best_weights=True)

def create_vgg16_manual(input_shape, num_classes):
    model = models.Sequential()

    #1
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    #2
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # 3
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    #  4
    model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    #  5
    model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))  

    return model

input_shape = (image_size, image_size, 3) 
num_classes = len(gestures)  
model = create_vgg16_manual(input_shape, num_classes)

for layer in model.layers[:-4]:  
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=15, batch_size=64, validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stopping, model_checkpoint])

model.save('models/mymodel.keras')
