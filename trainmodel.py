import os
import numpy as np
from PIL import Image
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


gestures = {'A': 'A', 'E': 'E', 'H': 'H', 'L': 'L', 'O': 'O'}
gestures_map = {'A': 0, 'E': 1, 'H': 2, 'L': 3, 'O': 4}
image_path = 'D:/XLA&TGMT/recognize_sign/Dataset'
models_path = 'models/saved_model.keras' 
image_size = 224


def process_image(path):
    img = Image.open(path)
    img = img.resize((image_size, image_size))
    img = np.array(img)
    return img


def process_data(X_data, y_data):
    X_data = np.array(X_data, dtype='float32')
    if X_data.shape[-1] != 3:  
        raise ValueError("Input images RGB")
    X_data /= 255  
    y_data = np.array(y_data)
    y_data = to_categorical(y_data)  
    return X_data, y_data


def walk_file_tree(image_path):
    X_data = []
    y_data = []
    for directory, subdirectories, files in os.walk(image_path):
        gesture_name = os.path.basename(directory)  
        if gesture_name in gestures_map:  
            for file in files:
                if not file.startswith('.'):  
                    path = os.path.join(directory, file)
                    y_data.append(gestures_map[gesture_name])
                    X_data.append(process_image(path))
        else:
            print(f"Warning: Gesture folder name not found for directory {directory}")  # In cảnh báo nếu không tìm thấy thư mục

    if len(X_data) == 0 or len(y_data) == 0:
        raise ValueError("Empty dataset found. Please make sure there are images and labels in the dataset.")
    
    X_data, y_data = process_data(X_data, y_data)
    return X_data, y_data


X_data, y_data = walk_file_tree(image_path)


X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=12, stratify=y_data)

model_checkpoint = ModelCheckpoint(filepath=models_path, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, restore_best_weights=True)


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))


x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(len(gestures), activation='softmax')(x)  # Số lớp đầu ra tương ứng với số ký hiệu
model = Model(inputs=base_model.input, outputs=predictions)


for layer in base_model.layers:
    layer.trainable = False


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stopping, model_checkpoint])

model.save('models/mymodel.keras')
