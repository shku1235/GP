# coding: utf-8

# In[1]:


import sys
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, losses
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Lambda, Cropping2D

import pandas as pd
from keras.utils import np_utils

from PIL import Image
import numpy as np

imgdim = 80

# In[13]:


model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=(imgdim, imgdim, 3), border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(leng, activation='softmax'))

# model = Sequential()
# model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(64,64,3)))
# model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
# model.add(Flatten())
# # model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# In[15]:


# diffrent working method

datagen = ImageDataGenerator(rescale=1. / 255)
x = []
y = []
for index, row in now.iterrows():
    #     print index
    name = row.values[0]
    #     y.append(res[index])

    img = Image.open("sg17402/shku/dataset/CaptionTraining2018/" + img[0] + ".jpg")
    img = img.resize((imgdim, imgdim)).convert('RGB')
    temp = np.array(img)
    #     print temp.shape
    x.append(temp)
#     yield (x, y)
# print x[0]
X = np.array(x)
print X.shape
print res.shape

model.fit_generator(datagen.flow(X, res), steps_per_epoch=550, epochs=1)

# In[29]:


T = []
name = 'PAMJ-22-72-g004'
img = Image.open("/Users/sharath/Desktop/GP/CaptionTraining2018/" + name + ".jpg")
img = img.resize((imgdim, imgdim)).convert('RGB')
T.append(np.array(img))
# print test.shape
X = np.array(T)
X.shape

# In[38]:


res = model.predict(X)

# In[39]:


res

# In[1]:


# datagen = ImageDataGenerator(
#     featurewise_center=True,
#     featurewise_std_normalization=True,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True)

# # compute quantities required for featurewise normalization
# # (std, mean, and principal components if ZCA whitening is applied)
# datagen.fit(x_train)

# # # fits the model on batches with real-time data augmentation:
# # model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
# #                     steps_per_epoch=len(x_train) / 32, epochs=epochs)

# # here's a more "manual" example
# for e in range(epochs):
#     print('Epoch', e)
#     batches = 0
#     for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
#         model.fit(x_batch, y_batch)
#         batches += 1
#         if batches >= len(x_train) / 32:
#             # we need to break the loop by hand because
#             # the generator loops indefinitely
#             break


# In[2]:


# def generate_arrays_from_file():

#     while 1:
#         f = open(path)
#         for line in f:
#             # create numpy arrays of input data
#             # and labels, from each line in the file
#             x, y = process_line(line)
#             img = load_images(x)
#             yield (img, y)
#         f.close()


# model.fit_generator(generate_arrays_from_file(),steps_per_epoch='10',epochs=10)


# log_dir = './tf-log/'
# tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# cbks = [tb_cb]

# target_dir = './models/'
# if not os.path.exists(target_dir):
#   os.mkdir(target_dir)
# model.save('./models/model.h5')
# model.save_weights('./models/weights.h5')


# In[3]:


# T = []
# name = 'aps-43-590-g001'
# img = Image.open("/Users/sharath/Desktop/GP/CaptionTraining2018/"+ name +".jpg")
# img = img.resize((64,64)).convert('RGB')
# T.append(np.array(img))
# # print test.shape
# X = np.array(T)
# X.shape


# In[4]:


# def generate_arrays_from_file():
#     for index, row in now.iterrows():
#         print index
#         x = row.values[0]
#         y = res[index]
#         img = Image.open("/Users/sharath/Desktop/GP/CaptionTraining2018/"+ x +".jpg")
#         T.append(np.array(img))
#         X = np.array(T)
#     yield (X, y)

# model.fit_generator(generate_arrays_from_file(),steps_per_epoch='1',epochs=1)


# log_dir = './tf-log/'
# tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# cbks = [tb_cb]

# target_dir = './models/'
# if not os.path.exists(target_dir):
#   os.mkdir(target_dir)
# model.save('./models/model.h5')
# model.save_weights('./models/weights.h5')


# In[ ]:


# from sklearn.metrics import accuracy_score
# from sklearn.ensemble import RandomForestClassifier
# #from sklearn.tree import DecisionTreeClassifier

# #run classification
# clf=RandomForestClassifier()
# clf.fit(X_train,y_train)

# #now, make predictions from the classifier
# y_predicts=clf.predict(X_val)
# acc = accuracy_score(y_val, y_predicts)
# print "Val acc: ", round(acc,3)

