#Imports
import os
import tensorflow as tf
keras = tf.keras
import pyodbc
from tensorflow.keras import layers, models
import cv2
import glob
import numpy as np

def Select_From_Where(conn,Element_required,table,condition_element,condition):
    cursor = conn.cursor()
    proc = "SELECT "+Element_required+" FROM "+table+" WHERE "+condition_element+" = "+condition+";"
    print(proc)
    cursor.execute(proc)
    for row in cursor:
        print(row[0])
        return row[0]
def READ(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT*FROM Customers')
    for row in cursor:
        print(row)


def USP(conn, PROC_NAME, Param_NAME, PARAM):
    cursor = conn.cursor()
    if (len(Param_NAME) == len(PARAM)):
        storedProc = "Exec " + PROC_NAME
        for i in range(len(Param_NAME)):
            if (i != len(Param_NAME) - 1):
                storedProc = storedProc + " " + Param_NAME[i] + " = " + str(PARAM[i]) + ","
            else:
                storedProc = storedProc + " " + Param_NAME[i] + " = " + str(PARAM[i])
        # Execute Stored Procedure With Parameters
        cursor.execute(storedProc)
        conn.commit()
        return True
    else:
        return "Length not equal"


def UPDATE(conn, DB_NAME, Param_NAME, PARAM):
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE " + DB_NAME + " SET " + Param_NAME[0] + " = " + str(PARAM[0]) + " WHERE " + Param_NAME[0] + " = " + str(
            PARAM[0]) + ";"
    )
    conn.commit()

#Function to rotate an image to a specific angle, can be used to increase dataset
def rotate(image, angle, center=None, scale=1.0):
    # Get image size
    (h, w) = image.shape[:2]

    # If the rotation center is not specified, set the image center as the rotation center
    if center is None:
        center = (w / 2, h / 2)

    # Perform rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # Return the rotated image#

#get images and resize to desired resolution
def get_images(location,format,lable,IMG_SIZE):
    image_array = []  # array which ll hold the images
    image_lable = []
    files = glob.glob(""+location+"*."+format+"")
    for myFile in files:
        
        image = cv2.imread(myFile)
        image_conv = tf.cast(image, tf.float32)
        image_conv = (image_conv / 127.5) - 1
        image_conv = tf.image.resize(image_conv, (IMG_SIZE, IMG_SIZE))
        image_array.append(image_conv)  # append each image to array
        image_lable.append(lable)

    # this will print the channel number, size, and number of images in the file
    print('image_array shape:', np.array(image_array).shape)
    #cv2.imshow('frame', image_array[0])
    #cv2.waitKey(0)
    return image_array,image_lable

#establish connection with DB
conn =pyodbc.connect(
    'Driver={SQL Server Native Client 11.0};'
    'Server=*********;'
    'Database=*******;'
    'Trusted_Connection=yes;')


className = ['PNEUMONIA','NORMAL']#Array of classes
image_cass = ['NORMAL_Train','Pneumonea_Train','NORMAL_Test','Pneumonea_Test','NORMAL_Val','Pneumonea_Val']#identifier for image paths in SQL Database

#get paths for each dataset in image_cass and store in image_paths
image_paths = []
for cass in image_cass:
    image_path = Select_From_Where(conn,'imagespath','dbo.image_table','imagename',"'"+cass+"'")
    image_paths.append(image_path)


#load train, test and validation images
#lableling NORMAL as 0 and PNEUMONIA as 1
image_array = []
lable_array = []

test_images = []
testlable_array = []

image_array_val = []
label_array_val = []

for x in range(2):
    IMG_SIZE = 128
    im_arr,lable_arr= get_images(""+image_paths[x]+"/","jpeg",x,IMG_SIZE)
    test_arr,testlable_arr = get_images(""+image_paths[x+2]+"/","jpeg",x,IMG_SIZE)
    im_arr_val, lable_arr_val = get_images("" + image_paths[x + 4] + "/", "jpeg", x, IMG_SIZE)
    image_array_val.extend(im_arr_val)
    label_array_val.extend(lable_arr_val)
    image_array.extend(im_arr)
    lable_array.extend(lable_arr)
    test_images.extend(test_arr)
    testlable_array.extend(testlable_arr)

#convert train images into numpy matrix
train_images_mat = np.array(image_array)
train_labels_mat = np.array(lable_array).transpose()

#convert test images into numpy matrix
#inflated_data = datagen(test_array)
#val_images = format_example(test_images,IMG_SIZE)
test_images_mat = np.array(test_images)
test_labels_mat = np.array(testlable_array).transpose()


#add convolutional neural network layers
model = models.Sequential()
model.add(layers.Conv2D(32, (2, 2), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (2, 2), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (2, 2), activation='relu'))
model.summary()  # let's have a look at our model so far

#add dense layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(len(className)))
model.summary()

#Compile model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#Save the model locarion in checkpoint_path
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

#Train the model
history = model.fit(train_images_mat, train_labels_mat, epochs=1,
                    validation_data=(test_images_mat, test_labels_mat),
                    callbacks=[cp_callback])

#test accuracy of test data set
test_loss, test_acc = model.evaluate(test_images_mat,  test_labels_mat, verbose=2)

#convert validation images into numpy matrix
predict_images_mat = np.array(image_array_val)
predict_labels_mat = np.array(label_array_val).transpose()
predictions = model.predict(predict_images_mat)

#Predict the validation data
predict =[]
for x in range(0,len(predictions),1):
    predict.append(np.argmax(predictions[x]))

print("done")
# pick an image to transform
