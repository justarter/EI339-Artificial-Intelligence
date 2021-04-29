# USAGE
# python train_keras_LeNet.py

# import the necessary packages
from pyimagesearch.models.keras_LeNet import LeNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from utils import *
from EI339_CN.processed.parse_data import *
import argparse
import cv2
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model",
	help="path to output model after training")
ap.add_argument("--batch_size", default=128)
ap.add_argument("--lr", default=1e-3)
ap.add_argument("--epochs", default=40)
args = vars(ap.parse_args())

# initialize the initial learning rate, number of epochs to train
# for, and batch size
INIT_LR = args["lr"]
EPOCHS = args["epochs"]
BS = args["batch_size"]

# grab the MNIST dataset
print("[INFO] accessing MNIST...")
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()#trainData 60000*28*28,testData 10000*28*28,trainLabels(60000,)(array)

# CN dataset
trainData_CN, trainLabels_CN, testData_CN, testLabels_CN = run()#train7286,test2034

# data augmentation
print("[INFO] data augmentation ...")
trainData_CN, trainLabels_CN = plus_augmentation(trainData_CN, trainLabels_CN)

# class 10-19 对应 中文 1-10
trainLabels_CN += 9
testLabels_CN += 9

# combine dataset
trainData = np.concatenate((trainData, trainData_CN), axis=0)
trainLabels = np.concatenate((trainLabels, trainLabels_CN), axis=0)
testData = np.concatenate((testData, testData_CN), axis=0)
testLabels = np.concatenate((testLabels, testLabels_CN), axis=0)

# add a channel (i.e., grayscale) dimension to the digits
trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))# 黑(0)底白(255)字,60000
testData = testData.reshape((testData.shape[0], 28, 28, 1))#10000

# pad for LeNet
trainData = np.pad(trainData, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')#32*32*1
testData = np.pad(testData, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

# scale data to the range of [0, 1]
trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0

# convert the labels from integers to vectors
le = LabelBinarizer()
trainLabels = le.fit_transform(trainLabels)
testLabels = le.transform(testLabels)

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR)
model = LeNet.build(width=32, height=32, depth=1, classes=20)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training network...")
history = LossHistory(INIT_LR, EPOCHS, BS)

H = model.fit(
	trainData, trainLabels,
	validation_data=(testData, testLabels),
	batch_size=BS,
	epochs=EPOCHS,
	verbose=1,
	callbacks=[history])

history.loss_plot('epoch')

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testData)
print(classification_report(
	testLabels.argmax(axis=1),
	predictions.argmax(axis=1),
	target_names=[str(x) for x in le.classes_]))

# serialize the model to disk
print("[INFO] serializing digit model...")
model.save("output/keras_LeNet{}{}{}.h5".format(BS, INIT_LR, EPOCHS), save_format="h5")
