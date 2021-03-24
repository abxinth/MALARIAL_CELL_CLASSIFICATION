import numpy as np
np.random.seed(1000)
import cv2
import os
from PIL import Image
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
tensorboard = TensorBoard(log_dir = 'logx_malarial')
#model.fit(X,Y,batch_size=32,epochs=3,callbacks=[tensorboard])

os.environ['KERAS_BACKEND'] = 'tensorflow'
SIZE = 64
INPUT_SHAPE = (SIZE,SIZE,3)

IMAGE_DIR = 'cell_images'
label = []
dataset =[]
parasitized = os.listdir(IMAGE_DIR + '/Parasitized/')
for i,image_name in enumerate(parasitized):
    if(image_name.split('.')[1] == 'png'):
        image = cv2.imread(IMAGE_DIR + '/Parasitized/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE,SIZE))
        dataset.append(np.array(image))
        label.append(0)
un_parasitized = os.listdir(IMAGE_DIR + '/Uninfected/')
for i,image_name in enumerate(un_parasitized):
    if(image_name.split('.')[1] == 'png'):
        image = cv2.imread(IMAGE_DIR + '/Uninfected/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE,SIZE))
        dataset.append(np.array(image))
        label.append(1)


###############################
print("data preparation")
X_train,X_test,y_train,y_test = train_test_split(dataset,keras.utils.to_categorical(np.array(label)),test_size=0.20,shuffle=True)

print("data preparation Done")
#################################
inp = keras.layers.Input(shape=INPUT_SHAPE)
conv1 = keras.layers.Conv2D(32,kernel_size=(3,3),activation='relu',padding='same')(inp)
pool1 = keras.layers.MaxPooling2D(pool_size=(2,2))(conv1)
norm1 = keras.layers.BatchNormalization(axis=-1)(pool1)
drop1 = keras.layers.Dropout(rate=0.2)(norm1)

conv2 = keras.layers.Conv2D(32,kernel_size=(3,3),activation='relu',padding='same')(drop1)
pool2 = keras.layers.MaxPooling2D(pool_size=(2,2))(conv2)
norm2 = keras.layers.BatchNormalization(axis=-1)(pool2)
drop2 = keras.layers.Dropout(rate=0.2)(norm2)

flat = keras.layers.Flatten()(drop2)
hidden1 = keras.layers.Dense(512,activation='relu')(flat)
norm3 = keras.layers.BatchNormalization(axis=-1)(hidden1)
hidden2 = keras.layers.Dense(256,activation='relu')(norm3)
norm4 = keras.layers.BatchNormalization(axis=-1)(hidden2)
drop4 = keras.layers.Dropout(rate=0.2)(norm4)
out = keras.layers.Dense(2,activation='sigmoid')(drop4)
model = keras.Model(inputs=inp,outputs=out)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
print(model.summary())
#######
history = model.fit(np.array(X_train),y_train,batch_size=64,verbose=1,epochs=3,validation_split=0.1,shuffle=False,callbacks=[tensorboard])
print(model.evaluate(np.array(X_test),np.array(y_test)))
#######
print("works fine")
