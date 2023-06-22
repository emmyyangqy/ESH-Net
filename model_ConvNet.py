###### ConvNet #####

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import openpyxl
import matplotlib.pyplot as plt
import keras.backend as K

from keras.models import Sequential
from keras.models import model_from_json
from keras.models import load_model, Model
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import BatchNormalization
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import categorical_accuracy
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.utils import plot_model

from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from mlxtend.plotting import plot_confusion_matrix

##### setting parameters
emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

label_map = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
names=['emotion','pixels','usage']

input_shape_length = 48


##### load images from files
base_dir = '/content'

global x_train, y_train, x_test, y_test

x_train = []
y_train = []
x_test = []
y_test = []

labels = os.listdir(os.path.join(base_dir, 'train'))
num_labels = len(labels)

desired_shape = (input_shape_length, input_shape_length)

for phase in ['train', 'test']:
    phase_dir = os.path.join(base_dir, phase)
    for label in os.listdir(phase_dir):
        label_dir = os.path.join(phase_dir, label)
        image_count = 0

        for image in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image)
                image_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image_data = image_data / 255.0

                if phase == 'train':
                    x_train.append(image_data)
                    y_train.append(label)
                else:
                    x_test.append(image_data)
                    y_test.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

y_train = to_categorical(y_train, num_labels)
y_test = to_categorical(y_test, num_labels)

x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)


##### CNN model architecture
num_features = 64
num_labels = 7
batch_size = 64
epochs = 50
width, height = 48, 48


##designing the cnn
# Initialising the CNN
model = Sequential()
# 1 - Convolution
model.add(Conv2D(128,(3,3), padding='same', input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

# 2nd Convolution layer
model.add(Conv2D(256,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

# 3rd Convolution layer
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

# 4th Convolution layer
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

# Flattening
model.add(Flatten())

# Fully connected layer 1st layer
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

# Fully connected layer 2nd layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(7, activation='softmax'))

opt = Adam(lr=0.0005)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#Training the model
h=model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          shuffle=True)


print("model fit is complete.")
print('\n')

# displan model plots
fig , ax = plt.subplots(1,2)
fig.set_size_inches(20, 8)

train_acc = h.history['accuracy']
train_loss = h.history['loss']
val_acc = h.history['val_accuracy']
val_loss = h.history['val_loss']

epochs = range(1, len(train_acc) + 1)

ax[0].plot(epochs , train_acc , 'g-o' , label = 'Training Accuracy')
ax[0].plot(epochs , val_acc , 'y-o' , label = 'Validation Accuracy')
ax[0].set_title('Model Training & Validation Accuracy')
ax[0].legend(loc = 'lower right')
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
ax[1].plot(epochs , val_loss , 'y-o' , label = 'Validation Loss')
ax[1].set_title('Model Training & Validation & Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")

plt.show()

# Print Train and Test accuracy & loss
print('Train loss & accuracy:', model.evaluate(x_train, y_train))
print('\n')
print('Test loss & accuracy:', model.evaluate(x_test, y_test))
print('\n')

# make prediction
y_test = np.argmax(y_test, axis=1)
yhat_test = np.argmax(model.predict(x_test), axis=1)


#### get confusion matrix
cm = confusion_matrix(y_test, yhat_test)
print("confusion_matrix:\n")
print(cm)

fig, ax = plot_confusion_matrix(conf_mat=cm,
                                show_normed=True,
                                show_absolute=False,
                                figsize=(8, 8))
plt.show()

# Calculate accuracy for each emotion category
accuracy_per_emotion = []
for i in range(num_labels):
    indices = np.where(y_test == i)[0]
    y_true_emotion = y_test[indices]
    y_pred_emotion = yhat_test[indices]
    accuracy_emotion = accuracy_score(y_true_emotion, y_pred_emotion)
    accuracy_per_emotion.append(accuracy_emotion)

# Print the percentages for each emotion category
print("Test Accuracy per Emotion Category:")
for i, emotion in enumerate(emotions.values()):
    print(f"{emotion}: {accuracy_per_emotion[i]:.2f}")


print("complete!")
