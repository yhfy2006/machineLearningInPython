
# coding: utf-8

# In[1]:

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, Flatten
from keras.models import Model
from IPython.display import SVG
from keras.utils.visualize_util import model_to_dot
from keras.callbacks import ModelCheckpoint
import os
import random
import keras
#from PIL import Image 


# In[2]:

ROW = 60
COL = 60
CHANNELS = 3

TRAIN_DIR = 'cat_dog_train/'
TEST_DIR = 'cat_dog_test/'

train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset
train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]

def readImg(imgFile):
    colorImg = cv2.imread(imgFile, cv2.IMREAD_COLOR)
    colorImg = cv2.cvtColor(colorImg, cv2.COLOR_BGR2RGB)
    colorImg = cv2.resize(colorImg, (ROW, COL))/255.0
    
    greyImg = cv2.imread(imgFile, cv2.IMREAD_GRAYSCALE)
    greyImg = cv2.cvtColor(greyImg,cv2.COLOR_GRAY2RGB)
    greyImg = cv2.resize(greyImg, (ROW, COL))/255.0
    
    return greyImg,colorImg
    
    

def generateDate(imgFiles):
    count = len(imgFiles)
    dataX = np.ndarray((count, ROW, COL,CHANNELS), dtype=float)
    dataY = np.ndarray((count, ROW, COL,CHANNELS), dtype=float)
    for i, image_file in enumerate(imgFiles):
        if(i%250==0):
            print("generated "+str(i)+" of "+str(count)+" images")
        gImg,cImg = readImg(image_file)
        dataX[i] = gImg
        dataY[i] = cImg  
    return dataX,dataY

# slice datasets for memory efficiency on Kaggle Kernels, delete if using full dataset
train_images = train_dogs[:3000] + train_cats[:3000]
random.shuffle(train_images)

dataX,dataY = generateDate(train_images)



# In[3]:

BaseLevel = ROW//2//2
input_img = Input(shape=(ROW,COL,CHANNELS))
x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Flatten()(x)
encoded = Dense(2000)(x)
oneD = Dense(BaseLevel*BaseLevel*128)(encoded)
fold = Reshape((BaseLevel,BaseLevel,128))(oneD)
x = UpSampling2D((2, 2))(fold)
x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)

decoded = Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
SVG(model_to_dot(autoencoder,show_shapes=True).create(prog='dot', format='svg'))


# In[4]:

# trainX =np.expand_dims(backtorgb, 0)
# trainy =np.expand_dims(colorImg, 0)


# In[ ]:
tensorBoardPath = '/home/yhfy2006/machineLearningInPython/AutoEncoder/logs'

tb_cb = keras.callbacks.TensorBoard(log_dir=tensorBoardPath, histogram_freq=1)
checkpoint = ModelCheckpoint(filepath="/home/yhfy2006/machineLearningInPython/AutoEncoder/logs/weights.hdf5", verbose=1, save_best_only=True)

cbks = [checkpoint]



autoencoder.fit(dataX, dataY,
                nb_epoch=400,
                batch_size=50,
                shuffle=True,
                verbose=1,
                validation_split=0.3,
                callbacks=cbks)

#autoencoder.save_weights("autoencoder.h5")

# In[49]:

test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]
test_images =  test_images[:10]

testImgRatios = []
for i,imgFile in enumerate(test_images):
    img = cv2.imread(imgFile)
    height, width, channels = img.shape
    ratio = float(width)/(height+width)
    print(ratio)
    testImgRatios.append(ratio)

testX,testY = generateDate(test_images)

predict = autoencoder.predict(testX)

for i,imageFile in enumerate(test_images):
    #predict = autoencoder.predict(testX)
    #plt.imshow(predict[0].astype(float))
#     print(testX[0].shape)
#     print(testX[0].reshape((-1,int(COL*testImgRatios[i]),3)))
#     print(int(ROW*COL*CHANNELS**testImgRatios[i]))
    plt.imshow(testX[i])
    plt.show()
    
    plt.imshow(predict[i].astype(float))
    plt.show()
    
    plt.imshow(testY[i])
    plt.show()

    


# In[144]:

autoencoder.summary()


# In[11]:

predict = autoencoder.predict(trainX)
get_ipython().magic('matplotlib inline')


# In[12]:

plt.imshow(predict[0].astype(float))


# In[79]:

predict[0].astype(int)


# In[88]:


print(trainy[0].astype(float))
plt.imshow(trainy[0][30])


# In[ ]:



