
# coding: utf-8

# In[12]:


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


# In[13]:


data = pd.read_csv('train_concepts.txt', delimiter=":", header=None)


# In[14]:


data.head()


# In[87]:


res = []
for index, row in data[:50].iterrows():
#     print index
    for i in row[1].split(';'):
#         print str(i)+';'+str(row[0])+';'+str(index)
        res.append(str(i)+';'+str(row[0])+';'+str(index))        


# In[88]:


final_results = pd.DataFrame(
    {
        "0":res,
    })
print final_results


# In[89]:


final_results.to_csv('Final_prediction1000.csv',  index = False, header=False)


# In[90]:


now = pd.read_csv('/Users/sharath/Desktop/GP/sharath/Final_prediction1000.csv', delimiter=";", header=None)
k = now[[2]].values
res = np_utils.to_categorical(k)
# print res[:5]
leng =  res.shape[1]
print len(now[0]), len(res)

dataset_size  = len(now[0])

# print now.head()

labels = pd.DataFrame(res)
# print labels.head()


result = pd.concat([now, labels], axis=1, join_axes=[now.index])
# print result.head()
# result.to_csv("preprocessed.txt", header=False)


# In[91]:


imgdim = 80


# In[92]:


model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape = (imgdim, imgdim,3),border_mode = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(500, activation = 'relu'))
model.add(Dense(leng, activation = 'softmax')) 

# model = Sequential()
# model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(64,64,3)))
# model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
# model.add(Flatten())
# # model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()


# In[93]:


# diffrent working method 

datagen = ImageDataGenerator(rescale=1. / 255)
x = []
y = []
for index, row in now.iterrows():
#     print index
    name = row.values[0] 
#     y.append(res[index])

    img = Image.open("/Users/sharath/Desktop/GP/CaptionTraining2018/"+ name +".jpg")
    img = img.resize((imgdim,imgdim)).convert('RGB')
    temp = np.array(img)
#     print temp.shape
    x.append(temp)
#     yield (x, y)
# print x[0]
X = np.array(x)
print X.shape
print res.shape

datagen.fit(X)
# model.fit_generator(datagen.flow(X,res, shuffle=True),steps_per_epoch=dataset_size, epochs=1)


# In[ ]:


for e in range(10):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in datagen.flow(X,res, shuffle=True ,batch_size=100):
        model.fit(x_batch, y_batch)
        batches += 1
        if batches >= len(x) / 32:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break


# In[80]:


