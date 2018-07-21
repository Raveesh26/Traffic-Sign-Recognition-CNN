
# coding: utf-8

# ### Loading the desired libraries

# In[2]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from keras.models import Sequential
from keras.layers import Flatten ,Dropout,Dense,Activation
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers import Conv2D,GlobalAveragePooling2D,MaxPooling2D,ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator


# ### Loading the data
# The data is stored in .p extension which is a pickle file.
# 
# `“Pickling” is the process whereby a Python object hierarchy is converted into a byte stream, and “unpickling” is the inverse operation, whereby a byte stream is converted back into an object hierarchy. Pickling (and unpickling) is alternatively known as “serialization”, “marshaling,” or “flattening” .`

# Test and Training data can be downloaded from here : https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581faac4_traffic-signs-data/traffic-signs-data.zip

# In[4]:

##Loading the Data
training_file ='train.p'
testing_file  ='test.p'

with open(training_file ,mode='rb') as f:
    train =pickle.load(f)

with open(testing_file ,mode='rb') as f:
    test =pickle.load(f)

x_train ,y_train =train['features'] ,train['labels']
x_test ,y_test   =test['features']  ,test['labels']


# In[9]:

print(x_train.shape ,y_train.shape)
print(x_test.shape ,y_test.shape)


# In[4]:

##This csv contains the lables name of the signs
signs_classID =pd.read_csv('signnames.csv')
signs_classID.iloc[:5]


# In[5]:

## unique classes stores the distinct labels
## unique_sign    stores the distinct classes labels
## unique_sign_counts stores the count of each label in the dataset

unique_classes,unique_sign ,unique_sign_counts=np.unique(y_train ,return_index =True ,return_counts=True)
unique_sign_counts


# ### Data Plotting and Exploration

# In[6]:

def plot_images():
    
    fig =plt.figure(figsize=(18,18))
    i=0
    for index in (unique_sign):
        #     ax.set_xticklabels([])
        #     ax.set_yticklabels([])
        #     ax.set_aspect('equal')
        img =x_train[index]
        plt.subplot(11,4,i+1)
        plt.imshow(img)
        plt.title( str(y_train[index])+'. '+signs_classID['SignName'][y_train[index]] )
        plt.axis('off')
        i =i+1
    plt.show()


# In[7]:

import matplotlib.image as img
plot_images()


# In[8]:

## plotting histogram

plt.figure(figsize=(15,5))
plt.bar(unique_classes,unique_sign_counts ,color='r' ,alpha=0.55)
plt.xlabel('label')
plt.ylabel('no. of distinct samples')
plt.xticks(unique_classes)
for a,b in zip(unique_classes ,unique_sign_counts):
    plt.text(a, b, str(b))
plt.tight_layout()
plt.show()


# ### Data Preprocessing

# Preprocessing includes normalization ,filtering noise , rescaling , dimension reduction of the images for better accuracy.
# 
# Here we will be using open cv2 library to normalize the images between 50 to 200  through MIN-MAX scaling.
# 
# 1.Normalization will mitigate differences due to light condition across the data set and will make the pixel intesity consistant.
# 
# 2.Image Blurring or (Image Smoothing) :
# 
# Bluring is done to remove noise/unwanted features. Median blur was choosen because it works good in removing salt and pepper noise and produces best results when tested in comparison with averaging, gaussian blur or bilateralFilter.
# 
# understanding source : https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html
# 

# In[10]:

## preprocessing and scaling of data
import cv2
def normailze_blur(img):
    img  =cv2.normalize(img, img, 50,200 ,cv2.NORM_MINMAX)
    cv2.medianBlur(img, 3)                                 ##here 30% noise is added to the original dataset and applied median blur on it
    return img


# In[11]:

for i in range(len(x_train)):
    normailze_blur(x_train[i])

for i in range(len(x_test)):
    normailze_blur(x_test[i])

## Plotting the normalized dataset
## clearly the are normalized on a given scale.
plot_images()


# ### Data Split into Train and Validation set

# In[13]:

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

x_train ,y_train = shuffle(x_train,y_train)
x_test ,y_test =shuffle(x_test,y_test)


# In[14]:

X_train , X_validation ,Y_train ,Y_validation =train_test_split(x_train,y_train,shuffle=True,random_state=0,test_size=0.3)


# #### One-hot encoding of the output as needed in the CNN network

# In[15]:

## one hot of the output
def one_hotencode(data):
    return np_utils.to_categorical(data,len(unique_classes))     # unique_classes here ,are 42


# In[16]:

Y_train = one_hotencode(Y_train)
Y_validation =one_hotencode(Y_validation)
y_test =one_hotencode(y_test)


# In[17]:

X_train.shape , X_validation.shape


# ### Defining the state of art (CNN) architecutre .
# 
# specifying the Conv2D layers , hidden ,flatten layers

# In[18]:

## specifying the layers
model =Sequential()

model.add(Conv2D(32,(5,5),strides=(1,1) ,input_shape=(32,32,3)))
model.add(Activation('relu'))
BatchNormalization(axis=-1)
model.add(Conv2D(32 ,(5,5)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
BatchNormalization(axis=-1)

model.add(Conv2D(64,(5,5)))
model.add(Activation('relu'))
BatchNormalization(axis =-1)

model.add(Conv2D(64 ,(5,5)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
BatchNormalization()

model.add(Dense(1024))
model.add(Activation('relu'))

BatchNormalization()
model.add(Dropout(0.5))

model.add(Dense(len(unique_classes)))
model.add(Activation('softmax'))


# In[19]:

model.compile(loss='categorical_crossentropy' ,optimizer=Adam() ,metrics=['accuracy'])


# In[20]:

model.summary()


# ### Data Augmentation
# Data Augmentation is certainly a better way to get most out of the few or small dataset.
# 
# Here Data Augmentation can certainly improve the accuracy of the network as rotating the image to a certain angle will not change the defined 'knowldege' or quality of the physical nature of the images.

# In[31]:

##Data augementation
##This network is paired using the Keras ImageDataGenerator to assist with image Augmentation through the learning process. 
##This can greatly improve the accuracy without the increase of labelled training examples.

gen = ImageDataGenerator(rotation_range=13 ,width_shift_range=0.12 ,shear_range=0.3 ,height_shift_range=0.12,zoom_range=0.2)
train_generator =gen.flow(X_train,Y_train,batch_size=32)


# In[32]:

validate_gen =ImageDataGenerator()

validate_generator =validate_gen.flow(X_validation ,Y_validation)


# ### Training the Model

# In[33]:

model.fit_generator(train_generator ,steps_per_epoch=27446//20 ,epochs=7 ,
                    validation_data =validate_generator,validation_steps =11763//80)


# In[34]:

x_test.shape ,y_test.shape


# #### Evaluating the accuracy metrics of the model

# In[39]:

accuracy =model.evaluate(x_test,y_test)


# In[40]:

accuracy


# In[42]:

print("Test Accuracy of the model --->" ,accuracy[1])


# In[72]:

np.unique(np.argmax(y_test,axis=1) ,return_index=True ,return_counts=True)


# In[53]:

model.predict_classes(x_test[0:50])


# In[87]:

## Plotting the forst 50 x_test images for the verification

y_pred_labels =model.predict_classes(x_test[0:50])

#Converting onehot-encoded labels back to their numerical data
y_test_labels =np.argmax(y_test ,axis=1)


# In[97]:

fig=plt.figure(figsize=(14,5))
for i in range(50):
    ax=fig.add_subplot(5,10,i+1 ,xticks=[] ,yticks=[])
    ax.imshow(x_train[i])
    ax.set_ylabel(y_pred_labels[i] ,color='black' if y_pred_labels[i]==y_test_labels[i] else 'red')
fig.suptitle('Predicted Labels; Incorrect Labels in Red' ,color='b' ,alpha=1.0)
plt.show


# In[ ]:



