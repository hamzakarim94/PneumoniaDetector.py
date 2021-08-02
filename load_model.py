import glob
import os

import cv2
import numpy as np
import pyodbc as pyodbc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


def Select_From_Where(conn,Element_required,table,condition_element,condition):
    cursor = conn.cursor()
    proc = "SELECT "+Element_required+" FROM "+table+" WHERE "+condition_element+" = "+condition+";"
    print(proc)
    cursor.execute(proc)
    for row in cursor:
        print(row[0])
        return row[0]


def get_images(location, format, lable, IMG_SIZE):
    image_array = []  # array which ll hold the images
    image_lable = []
    files = glob.glob("" + location + "*." + format + "")
    for myFile in files:
        image = cv2.imread(myFile)
        image_conv = tf.cast(image, tf.float32)
        image_conv = (image_conv / 127.5) - 1
        image_conv = tf.image.resize(image_conv, (IMG_SIZE, IMG_SIZE))
        image_array.append(image_conv)  # append each image to array
        image_lable.append(lable)

    # this will print the channel number, size, and number of images in the file
    print('image_array shape:', np.array(image_array).shape)
    # cv2.imshow('frame', image_array[0])
    # cv2.waitKey(0)
    return image_array, image_lable


def create_model(className):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (2, 2), activation='relu', input_shape=(128, 128, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (2, 2), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (2, 2), activation='relu'))
    model.summary()  # let's have a look at our model so far

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(len(className)))
    model.summary()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

className = ['PNEUMONIA','NORMAL']
image_cass = ['NORMAL_Train','Pneumonea_Train','NORMAL_Test','Pneumonea_Test','NORMAL_Val','Pneumonea_Val']
image_paths = []

conn =pyodbc.connect(
    'Driver={SQL Server Native Client 11.0};'
    'Server=*******;'
    'Database=*****;'
    'Trusted_Connection=yes;')

for cass in image_cass:
    image_path = Select_From_Where(conn,'imagespath','dbo.image_table','imagename',"'"+cass+"'")
    image_paths.append(image_path)

#Load test images
image_array = []
lable_array = []

test_images = []
testlable_array = []


for x in range(2):
    IMG_SIZE = 128
    test_arr,testlable_arr = get_images(""+image_paths[x+2]+"/","jpeg",x,IMG_SIZE)
    test_images.extend(test_arr)
    testlable_array.extend(testlable_arr)

test_images_mat = np.array(test_images)
test_labels_mat = np.array(testlable_array).transpose()

#create an untrained model
untrained_model = create_model(className)
checkpoint_path = "training_1/cp.ckpt"

#load weights into the untrained model with the weights stored in checkpoint path
untrained_model.load_weights(checkpoint_path)

#test accuraacy of the test dataset
loss, acc = untrained_model.evaluate(test_images_mat, test_labels_mat, verbose=2)
print(loss)
print(acc)