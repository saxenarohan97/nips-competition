### classifier in keras
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D,AveragePooling2D ,Conv2DTranspose,ZeroPadding2D,UpSampling2D,BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.preprocessing import normalize
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import keras
from keras.utils.BilinearUpSampling import *
from scipy import ndimage
from keras import regularizers
import cPickle as pickle
import numpy as np
import cv2
import shutil
np.set_printoptions(threshold='nan')

##Don't forget to change name and index

print('loading data....')
#fileindex = 't200-mangofull'
#x_train = pickle.load(open('./train_data/x_train' + fileindex + '.p','rb'))
#y_train = pickle.load(open('./train_data/y_train' + fileindex1 + '.p','rb'))

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

print(np.shape(x_train))
print(np.shape(y_train))
i =100
print('data loaded....')

name = 'FCN_model'

try:
	shutil.rmtree('./log/' + name )
	print('Deleted logdir/' + name +  '...')
except:
	print('No directory named log/'+ name + '...')


nb_epoch = 250

n = np.shape(x_train)[1]
model = Sequential()


#Convolution Layer
#border_mode 'same' adds one zero padding layer and works only when convolution has stride one

#Block 1 
model.add(Convolution2D(32,(3,3), input_shape=[n,n,3],padding="same",name="block1_conv1"))	
model.add(Activation('relu'))
model.add(Convolution2D(32, (3,3),padding="same",name="block1_conv2"))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2),name="block1_pool1"))

#block 2
model.add(Convolution2D(64, (3,3),padding="same",name="block2_conv1"))
model.add(Activation('relu'))
model.add(Convolution2D(64, (3,3),padding="same",name="block2_conv2"))
model.add(AveragePooling2D(pool_size=(2,2),name="block2_pool1"))

#block 3
model.add(Convolution2D(128, (3,3),padding="same",name="block3_conv1"))
model.add(Activation('relu'))
model.add(Convolution2D(128, (3,3),padding="same",name="block3_conv2"))
model.add(Activation('relu'))
model.add(Convolution2D(128, (3,3),padding="same",name="block3_conv3"))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2),name="block3_pool1"))

#block 4
model.add(Convolution2D(256, (3,3),padding="same",name="block4_conv1"))
model.add(Activation('relu'))
model.add(Convolution2D(256, (3,3),padding="same",name="block4_conv2"))
model.add(Activation('relu'))
model.add(Convolution2D(256, (3,3),padding="same",name="block4_conv3"))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2),name="block4_pool2"))

#block 5
model.add(Convolution2D(512, (3,3),padding="same",name="block5_conv1"))
model.add(Activation('relu'))
model.add(Convolution2D(512, (3,3),padding="same",name="block5_conv2"))
model.add(Activation('relu'))
model.add(Convolution2D(512, (3,3),padding="same",name="block5_conv3"))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2),name="block5_pool1"))

#FCN

model.add(Convolution2D(512,(6,6),name="FCN_1",kernel_regularizer=regularizers.l2(0.)))
model.add(Activation('relu'))
model.add(Convolution2D(512,(1,1),name="FCN_2",kernel_regularizer=regularizers.l2(0)))
model.add(Activation('relu'))
model.add(Convolution2D(1,(1,1),name="FCN_3",kernel_regularizer=regularizers.l2(0)))
model.add(Activation('sigmoid',name="sigmoid"))
#model.add(Activation('relu'))

#Upsampling
#model.add(BilinearUpSampling2D(target_size = (512,512)))
#model.add(UpSampling2D(size=(200,200)))
#Deconvolutional Layer
#model.add(Conv2DTranspose(1,(25,25),strides=(1,1),name="deconv_1"))
#model.add(Conv2DTranspose(1,(2,2),strides=(2,2),name="deconv_2"))
#model.add(Conv2DTranspose(1,(4,4),strides=(4,4),name="deconv_3"))
#model.add(Activation('sigmoid',name="sigmoid"))
#model.add(Activation('sigmoid',name="sigmoid"))
model.add(UpSampling2D(size=(200,200)))
model.summary()

sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)
model.compile(loss="mse",
               optimizer=sgd,
               metrics=['accuracy'])


tb = keras.callbacks.TensorBoard(log_dir='./log/' + name , histogram_freq=5,write_graph=True, write_images=False)

lr_decay = keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.1, patience=2,min_lr=0.00000001)

save_model = keras.callbacks.ModelCheckpoint('./models/'+ name +'.h5',monitor='val_loss',period=2)

model.fit(x_train,y_train,batch_size = 8, epochs=nb_epoch,callbacks = [tb,lr_decay,save_model])

