#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import  Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
#from tensorflow.keras.optimizers import RMSprop


# In[2]:


data = 'DATASET/'
train_dir = os.path.join(data, 'TRAIN')
test_dir = os.path.join(data, 'TEST')


# In[3]:


# augmentation
train_generator = ImageDataGenerator(rescale=1/255, rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest')

test_generator = ImageDataGenerator(rescale=1/255)


# In[9]:


modele = Sequential(layers=[
    Conv2D(filters=64, 
           kernel_size=(3,3), 
           padding="same",activation='relu', 
           input_shape=(224,224,3)),
    
    MaxPool2D(pool_size=(2,2)),
    
    Conv2D(filters=32, 
           kernel_size=(3,3), 
           activation='relu', ),
           
    MaxPool2D(pool_size=(2,2)),
           
    Conv2D(filters=128, 
           kernel_size=(3,3), 
           activation='relu', ),
           
    MaxPool2D(pool_size=(2,2)),
           
    Conv2D(filters=128, 
           kernel_size=(3,3), 
           activation='relu', ),
          
    MaxPool2D(pool_size=(2,2)),
    
    #flaten
    Flatten(),
    Dense(units=128, activation='relu'),
    Dense(units=64, activation="relu"),
    Dense(units=32, activation = "relu"),
    Dense(1, activation='sigmoid')
])


# In[10]:


train_image_generator = train_generator.flow_from_directory(train_dir, 
                                                            target_size= (224,224), 
                                                            batch_size = 32,
                                                            class_mode='binary'
                                                           )

test_image_generator = test_generator.flow_from_directory(test_dir, 
                                                            target_size= (224,224), 
                                                            batch_size = 32,
                                                            class_mode='binary'
                                                           )


# In[11]:


model_chepoint = ModelCheckpoint(filepath="model_jb.keras", 
                                 save_best_only=True,
                                monitor = 'val_accuracy', 
                                 mode='max')

stop = EarlyStopping(monitor='val_accuracy', 
                     patience=4)


# In[12]:


modele.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate = 0.0001),
               loss = 'binary_crossentropy', 
               metrics = ['accuracy'])


# In[ ]:


h2 = modele.fit(train_image_generator, epochs=30, 
           validation_data=test_image_generator,
          callbacks=[model_chepoint, stop])


# In[47]:


import matplotlib.pyplot as plt


# In[49]:


plt.plot(h2.history['val_accuracy'])
plt.plot(h2.history['accuracy'])


# In[12]:





# In[13]:





# In[14]:





# In[ ]:





# In[ ]:





# In[ ]:




