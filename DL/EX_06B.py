# Import necessary packages
import matplotlib.pyplot as plt
import numpy as np  
import pandas as pd  
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_path = r"D:\clg\DL\chest_xray\train"
test_path = r"D:\clg\DL\chest_xray\test"
valid_path = r"D:\clg\DL\chest_xray\val"
batch_size = 16
img_height = 99
img_width = 128
image_gen = ImageDataGenerator(
    rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)
test_data_gen = ImageDataGenerator(rescale=1./255)
train = image_gen.flow_from_directory(
    train_path, target_size=(img_height, img_width), color_mode='grayscale', class_mode='binary', batch_size=batch_size
)

test = test_data_gen.flow_from_directory(
    test_path, target_size=(img_height, img_width), color_mode='grayscale',
    shuffle=False,
    class_mode='binary', batch_size=batch_size
)

valid = test_data_gen.flow_from_directory(
    valid_path, target_size=(img_height, img_width), color_mode='grayscale', class_mode='binary', batch_size=batch_size
)
plt.figure(figsize=(12, 12))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    for X_batch, Y_batch in train:
        image = X_batch[0]
        label = Y_batch[0]
        dic = {0: 'NORMAL', 1: 'PNEUMONIA'}
        plt.title(dic.get(label))
        plt.axis('off')
        plt.imshow(np.squeeze(image), cmap='gray', interpolation='nearest')
        break
plt.tight_layout()
plt.show()
