#!/usr/bin/env python
# coding: utf-8

# # Watersheds Segmentation <a href="https://mybinder.org/v2/gh/InsightSoftwareConsortium/SimpleITK-Notebooks/master?filepath=Python%2F32_Watersheds_Segmentation.ipynb"><img style="float: right;" src="https://mybinder.org/badge_logo.svg"></a>

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import SimpleITK as sitk
from myshow import myshow, myshow3d
import numpy as np

# Download data to work on
get_ipython().run_line_magic('run', 'update_path_to_download_script')
from downloaddata import fetch_data as fdata

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import random


# In[ ]:


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
print(f)


# In[ ]:


reader = sitk.ImageSeriesReader()

def get_segementation_image(name):
    dicom_names = reader.GetGDCMSeriesFileNames(name)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    segmentation(image)
    return sitk.GetArrayFromImage(image)

def segmentation(image):
    image = sitk.GradientMagnitude(image)
    return sitk.MorphologicalWatershed(image, level=25, markWatershedLine=True, fullyConnected=False)

def get_training_data(arr):
    training_data = []
    first = True
    for fn in arr:
        path = str(fn)
        try:
            image = get_segementation_image(path)
        except:
            continue
        if len(image) > 50:
            if first:
                training_data = [random.sample(list(image), 50)]
                first = False
            else:
                training_data = np.concatenate((training_data, [random.sample(list(image), 50)]))
        
    return training_data


# In[ ]:


training_data_covid = get_training_data(f)


# In[ ]:


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
print(np.shape(training_data_hl_1))


# In[ ]:


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
print(np.shape(training_data_hl_2))


# In[ ]:


print(np.shape(training_data_hl_1))
print(np.shape(training_data_covid))
print(np.shape(training_data_hl_2))
training_data = np.concatenate((training_data_hl_1, training_data_covid, training_data_hl_2))
print(np.shape(training_data))


# In[ ]:


labels = np.concatenate(([0] * len(training_data_hl_1), [1] * len(training_data_covid), [0] * len(training_data_hl_2)))
print(np.shape(labels)[0])
training_data = training_data/255


# In[ ]:


split_len = int(np.shape(labels)[0] * 0.8)


# In[ ]:


def build_model():
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
    return model
    
model = build_model()


# In[ ]:


callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./model/cp-{epoch:04d}.ckpt',
                                                 save_weights_only=True,
                                                 verbose=1)


# In[ ]:


history = model.fit(training_data[:split_len], labels[:split_len], epochs=20, batch_size=5, validation_data=(training_data[split_len:], labels[split_len:]), callbacks=callback)


# In[ ]:


from matplotlib import pyplot
pyplot.plot(history.history['loss'], label='train_loss')
pyplot.plot(history.history['val_loss'], label='test_loss')
pyplot.legend()
pyplot.show()


# In[ ]:


prediction = model.predict(training_data[split_len:])


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix
def flattt(arr):
    output = []
    for aa in arr:
        output.append(aa)
    return output
def flattt_pred(arr):
    output = []
    for aa in arr:
        if aa > 0:
            output.append(1)
        else:
            output.append(0)
    return output

print(classification_report(labels[split_len:], flattt_pred(prediction), target_names=['0', '1']))
print(confusion_matrix(labels[split_len:],flattt_pred(prediction)))


# # The code for evaluating dataset and watershed level

# In[ ]:


# list
# print(data['labelGroups'][2])
for i in range(0, len(data['datasets'])):
    for j in range(0, len(data['datasets'][i]['annotations'])):
        if '1.2.826.0.1.3680043.10.474.419639.312580455409613733097488204614' == data['datasets'][i]['annotations'][j].get('StudyInstanceUID', 0):
            print(data['datasets'][i]['annotations'][j]['labelId'])
    


# In[ ]:


import json
# Opening JSON file
f = open('MIDRC-RICORD-1a_annotations_labelgroup_all_2020-Dec-8 2.json')
 
# returns JSON object as
# a dictionary
data = json.load(f)
 
# Iterating through the json
# list
for i in data:
    print(i)
 
# Closing file
f.close()


# In[ ]:


data['datasets'][0]['annotations'][0]['SeriesInstanceUID']


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import SimpleITK as sitk
from myshow import myshow, myshow3d
import numpy as np

# Download data to work on
get_ipython().run_line_magic('run', 'update_path_to_download_script')
from downloaddata import fetch_data as fdata

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import random


# In[ ]:


reader = sitk.ImageSeriesReader()

dicom_names = reader.GetGDCMSeriesFileNames('./2.000000-ROUTINE CHEST NON-CON-97100/')
reader.SetFileNames(dicom_names)
imagess = reader.Execute()
myshow(imagess)


# In[ ]:


feature_img = sitk.GradientMagnitude(imagess)
myshow(feature_img)


# In[ ]:


ws_img = sitk.MorphologicalWatershed(feature_img, level=10, markWatershedLine=True, fullyConnected=False)
myshow(sitk.LabelToRGB(ws_img), "Watershed Over Segmentation")


# In[ ]:


ws_img = sitk.MorphologicalWatershed(feature_img, level=30, markWatershedLine=True, fullyConnected=False)
myshow(sitk.LabelToRGB(ws_img), "Watershed Over Segmentation")


# In[ ]:


ws_img = sitk.MorphologicalWatershed(feature_img, level=50, markWatershedLine=True, fullyConnected=False)
myshow(sitk.LabelToRGB(ws_img), "Watershed Over Segmentation")


# In[ ]:


ws_img = sitk.MorphologicalWatershed(feature_img, level=20, markWatershedLine=True, fullyConnected=False)
myshow(sitk.LabelToRGB(ws_img), "Watershed Over Segmentation")


# In[ ]:




