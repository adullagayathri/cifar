#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import cifar10
import keras
from keras.models import Sequential 
from keras.layers import Conv2D,MaxPooling2D,AveragePooling2D,Dense,Flatten,Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard


# In[2]:


(X_train, y_train), (X_test, y_test) = cifar10.load_data() 


# In[3]:


X_train.shape


# In[4]:


X_test.shape


# In[5]:


y_train.shape


# In[6]:


y_train[1]


# In[7]:


X_train[0]
plt.imshow(X_train[1])


# In[8]:


y_test.shape


# In[9]:


for i in range(0,9):
    plt.subplot(330+1+i)
    plt.imshow(X_train[i])
plt.show()


# In[10]:


for i in range(0,15):
    #plt.subplot(330+2+i)
    plt.imshow(X_train[i])
    plt.show()


# In[11]:


X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
number_cat=10


# In[12]:


y_train


# In[13]:


y_train=keras.utils.to_categorical(y_train,number_cat)


# In[14]:


y_train


# In[15]:


y_test=keras.utils.to_categorical(y_test,number_cat)


# In[16]:


y_test


# In[17]:


X_train=X_train/255
X_test=X_test/255


# In[18]:


#X_train


# In[19]:


X_train.shape


# In[20]:


Input_shape = X_train.shape[1:]


# In[21]:


Input_shape


# In[22]:


model=Sequential()
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',input_shape=Input_shape))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu', padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.4))

model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu', padding='same'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu', padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(units=10,activation='softmax'))


# In[26]:


model.compile(loss='categorical_crossentropy',metrics=['accuracy'])


# In[27]:


history=model.fit(X_train,y_train,validation_split=0.2 ,batch_size=128,epochs=150,shuffle=True)


# In[28]:


evaluation=model.evaluate(X_test,y_test)
print('Test Accuracy: {}'.format(evaluation[1]))


# In[29]:


prediction = model.predict([X_test])
prediction


# In[30]:


print('Probabilities: ', prediction[10])
print('\n')
print('Prediction: ', np.argmax(prediction[10]))


# In[31]:


y_test=y_test.argmax(1)


# In[32]:


y_test


# In[33]:


L=8
W=8
fig,axes=plt.subplots(L,W,figsize=(12,12))
axes=axes.ravel()

for i in np.arange(0,L*W):
    axes[i].imshow(X_test[i])
    axes[i].set_title('Prediction= {}\nTrue={}'.format(prediction[i],y_test[i]))
    axes[i].axis('off')
    plt.subplots_adjust(wspace=1)



# In[34]:


plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()


# In[35]:


import pandas as pd
history_df = pd.DataFrame({
    'Epoch': range(1, len(history.history['accuracy']) + 1),
    'Training Accuracy': history.history['accuracy'],
    'Validation Accuracy': history.history['val_accuracy']
})


history_df.to_excel('accuracy_cifar.xlsx', index=False)


# In[36]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import plot_model
import pydot
import graphviz
from IPython.display import SVG


# In[37]:


plot_model(model, to_file='model_cifar.png', show_shapes=True, show_layer_names=True)


# In[38]:


img = plt.imread('model_cifar.png')
plt.imshow(img)
plt.axis('off')
plt.show()


# In[ ]:




