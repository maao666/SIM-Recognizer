from skimage import io, transform
from sklearn.model_selection import train_test_split
import glob
import os
import numpy as np
import keras

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


path = './dataset/'

w = 44
h = 68
c = 3


def read_image(path):
    img_contents = [path+x for x in os.listdir(path) if os.path.isdir(path+x)]

    img_contents.sort()
    print(img_contents)

    imgs = []
    labels = []
    for idx, folder in enumerate(img_contents):
        print(idx, folder)

        for im in glob.glob(folder + '/*.bmp'):
            img = io.imread(im)

            img = transform.resize(img, (w, h))
            imgs.append(img)
            labels.append(idx)

    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)

data, label = read_image(path)

print(data[0])
print(label)
print(len(label))


train_x, test_x, train_y, test_y = train_test_split(
    data, label, test_size=0.33, random_state=123)

print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)

print(train_y[0])
train_y = keras.utils.to_categorical(train_y, num_classes=16)
test_y = keras.utils.to_categorical(test_y, num_classes=16)
print(train_y[0])


classifier = Sequential()
classifier.add(Conv2D(filters=32, kernel_size=(3, 3),
                      padding='same', input_shape=(44, 68, 3),
                      activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())

classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=16, activation='softmax'))

classifier.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

classifier.fit(train_x, train_y, batch_size=30,
               epochs=30, validation_data=(test_x, test_y))
classifier.save('sim_iccid_model.hdf5')
print('Model saved')
