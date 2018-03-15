
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
# 

imgdim = 640

# In[10]:


model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=(imgdim, imgdim, 3), border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(leng, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

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



