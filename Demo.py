#!/usr/bin/env python
# coding: utf-8

# # Watersheds Segmentation <a href="https://mybinder.org/v2/gh/InsightSoftwareConsortium/SimpleITK-Notebooks/master?filepath=Python%2F32_Watersheds_Segmentation.ipynb"><img style="float: right;" src="https://mybinder.org/badge_logo.svg"></a>

# In[1]:

import matplotlib.pyplot as plt
import SimpleITK as sitk
from myshow import myshow, myshow3d
import numpy as np

# Download data to work on
from downloaddata import fetch_data as fdata

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import random

import warnings
warnings.filterwarnings("ignore")


# In[2]:


from os import walk

f = []
for (dirpath1, dirnames1, filenames1) in walk('./data/MIDRC-RICORD-1A/'):
    for dir1 in dirnames1:
        for (dirpath2, dirnames2, filenames2) in walk('./data/MIDRC-RICORD-1A/' + dir1):
            for dir2 in dirnames2:
                for (dirpath3, dirnames3, filenames3) in walk('./data/MIDRC-RICORD-1A/' + dir1 + '/' + dir2):
                    for dir3 in dirnames3:
                        f.append('./data/MIDRC-RICORD-1A/' + dir1 + '/' + dir2 + '/' + dir3)
    
    break


# In[3]:


reader = sitk.ImageSeriesReader()

def get_segementation_image(name):
    dicom_names = reader.GetGDCMSeriesFileNames(name)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
#     segmentation(image)
    return sitk.GetArrayFromImage(image)

def segmentation(image):
    image = sitk.GradientMagnitude(image)
    return sitk.MorphologicalWatershed(image, level=25, markWatershedLine=True, fullyConnected=False)

def get_training_data(arr):
#     print(arr)
    training_data = []
    first = True
    for fn in arr:
        path = str(fn)
#         print(path)
        try:
            image = get_segementation_image(path)
        except:
            continue
#         print(len(image))
        if len(image) > 50:
            if first:
                training_data = [random.sample(list(image), 50)]
                first = False
            else:
                training_data = np.concatenate((training_data, [random.sample(list(image), 50)]))
        
    return training_data


# In[4]:


print('Loading Covid lung data and preprocessing...')
training_data_covid = get_training_data(f)


# In[6]:


print('Loading healthy lung data and preprocessing...')

f_hl_1 = []
for (dirpath1, dirnames1, filenames1) in walk('./health_lung_1_dir/'):
    for dir1 in dirnames1:
        for (dirpath2, dirnames2, filenames2) in walk('./health_lung_1_dir/' + dir1):
            for dir2 in dirnames2:
                for (dirpath3, dirnames3, filenames3) in walk('./health_lung_1_dir/' + dir1 + '/' + dir2):
                    for dir3 in dirnames3:
                        f_hl_1.append('./health_lung_1_dir/' + dir1 + '/' + dir2 + '/' + dir3)
    
    break

training_data_hl_1 = get_training_data(f_hl_1)


# In[7]:


f_hl_2 = []
for (dirpath1, dirnames1, filenames1) in walk('./health_lung_2_dir/'):
    for dir1 in dirnames1:
        for (dirpath2, dirnames2, filenames2) in walk('./health_lung_2_dir/' + dir1):
            for dir2 in dirnames2:
                for (dirpath3, dirnames3, filenames3) in walk('./health_lung_2_dir/' + dir1 + '/' + dir2):
                    for dir3 in dirnames3:
                        f_hl_2.append('./health_lung_2_dir/' + dir1 + '/' + dir2 + '/' + dir3)
    
    break

training_data_hl_2 = get_training_data(f_hl_2)


# In[8]:

training_data = np.concatenate((training_data_hl_1, training_data_covid, training_data_hl_2))
print('the shape of evaulation dataset ', np.shape(training_data))


# In[9]:


labels = np.concatenate(([0] * len(training_data_hl_1), [1] * len(training_data_covid), [0] * len(training_data_hl_2)))
training_data = training_data/255


# In[10]:


split_len = int(np.shape(labels)[0] * 0.8)


# In[11]:


model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(50, 512, 512)))
model.add(layers.Dropout(0.4))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['binary_accuracy'])


# In[12]:

print("Evaluate untrained model...")
loss, acc = model.evaluate(training_data, labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))


print('Loading trained model weight')
model.load_weights('./model/cp-0006.ckpt')

print("Evaluate Restored model...")
loss, acc = model.evaluate(training_data, labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


reader = sitk.ImageSeriesReader()

print('Enter image path:')
x = input()
dicom_names = reader.GetGDCMSeriesFileNames(x) # ./2.000000-ROUTINE CHEST NON-CON-97100/
reader.SetFileNames(dicom_names)
imagess = reader.Execute()
myshow(imagess)

y = input()
dicom_nameh = reader.GetGDCMSeriesFileNames(y) # ./healthy_lung/
reader.SetFileNames(dicom_nameh)
imageh = reader.Execute()
myshow(imageh)

input_data = np.concatenate(([random.sample(list(sitk.GetArrayFromImage(imageh)), 50)], [random.sample(list(sitk.GetArrayFromImage(imagess)), 50)]))

prediction = model.predict(input_data)

print(prediction)

if prediction[0] >= 0.5:
    print(x, 'may be positive')
else: 
    print(x, 'may be negative')
    
if prediction[1] >= 0.5:
    print(y, 'may be positive')
else: 
    print(y, 'may be negative')

