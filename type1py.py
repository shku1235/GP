
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


data = pd.read_csv('train_concepts.txt', delimiter=":", header=None)

# In[3]:


data.head()

# In[13]:


res = []
for index, row in data[:10000].iterrows():
    #     print index
    for i in row[1].split(';'):
        #         print str(i)+';'+str(row[0])+';'+str(index)
        res.append(str(i) + ';' + str(row[0]) + ';' + str(index))

    # In[14]:

final_results = pd.DataFrame(
    {
        "0": res,
    })
print final_results

# In[15]:


final_results.to_csv('Final_prediction1000.csv', index=False, header=False)

# In[16]:


now = pd.read_csv('Final_prediction1000.csv', delimiter=";", header=None)
k = now[[2]].values
res = np_utils.to_categorical(k)
# print res[:5]
leng = res.shape[1]
print len(now[0]), len(res)

dataset_size = len(now[0])

# print now.head()

labels = pd.DataFrame(res)
# print labels.head()


result = pd.concat([now, labels], axis=1, join_axes=[now.index])
from sklearn.utils import shuffle

result = shuffle(result)
print result.head()

result.to_csv("preprocessed.txt", header=False)

# In[18]:


imgdim = 640

# In[10]:


model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=(imgdim, imgdim, 3), border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(leng, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['f'])

model.summary()


# In[12]:


def generate_arrays_from_file(path):
    def process_line(row):
        temp = row.split(',')
        #         print temp[0]
        return temp[1:2], temp[4:]

    def load_images(img):
        #         print img
        img = Image.open("sg17402/shku/dataset/CaptionTraining2018/" + img[0] + ".jpg")
        img = img.resize((imgdim, imgdim)).convert('RGB')
        T = (np.expand_dims(img, axis=0))
        return T

    while 1:
        f = open(path)
        for line in f:
            x, y = process_line(line)
            img = load_images(x)
            y = np.expand_dims(y, axis=0)
            #             print img.shape, y.shape
            yield (img, y)
        f.close()


path = 'sg17402/shku/GP/preprocessed.txt'

model.fit_generator(generate_arrays_from_file(path), steps_per_epoch=dataset_size, epochs=1)

# In[116]:


# T = []
# name = 'MGG3-4-599-g001'
# img = Image.open("/Users/sharath/Desktop/GP/CaptionTraining2018/" + name + ".jpg")
# img = img.resize((imgdim, imgdim)).convert('RGB')
# T.append(np.array(img))
# X = np.array(T)
# X.shape
#
# # In[117]:
#
#
# model.predict_classes(X)


# In[ ]:
#
#
# def test_genarator(track_index):
#     track_index = track_index
#     X = []
#     y = []
#     print  track_index
#     for index, row in testdata[track_index:track_index + plus_track_index].iterrows():
#         #         print row.index
#         x = row.values[0]
#         label = row.values[3:]
#         x = row.values[0]
#         y.append(label)
#         img = Image.open("/Users/sharath/Desktop/GP/CaptionTraining2018/" + x + ".jpg")
#         img = img.resize((imgdim, imgdim)).convert('RGB')
#         T = (np.array(img))
#         X.append(T)
#     #         track_index =  index
#     X_train = np.array(X)
#     y_train = np.array(y)
#     #     print X_train.shape, y_train.shape
#     return X_train, y_train

# In[ ]:


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


# In[ ]:


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


# In[1]:


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

