# python train_numpy_LeNet.py

import numpy as np
import matplotlib.pyplot as plt
from layers.base_conv import Conv2D
from layers.fc import FullyConnect
from layers.pooling import MaxPooling
from layers.softmax import Softmax
from layers.relu import Relu
from tensorflow.keras.datasets import mnist
import argparse
import pickle
import time

class LeNet():
    def __init__(self, batch_size):
        self.classes = 10
        self.batch_size = batch_size
        self.conv1 = Conv2D([batch_size, 32, 32, 1], 6, 5, 1)
        self.relu1 = Relu(self.conv1.output_shape)
        self.pool1 = MaxPooling(self.relu1.output_shape)
        self.conv2 = Conv2D(self.pool1.output_shape, 16, 5, 1)
        self.relu2 = Relu(self.conv2.output_shape)
        self.pool2 = MaxPooling(self.relu2.output_shape)
        self.fc1 = FullyConnect(self.pool2.output_shape, 120)
        self.fc2 = FullyConnect(self.fc1.output_shape, 84)
        self.fc3 = FullyConnect(self.fc2.output_shape, self.classes)
        self.sf = Softmax(self.fc3.output_shape)

    def forward(self, img):
        conv1_out = self.relu1.forward(self.conv1.forward(img))
        pool1_out = self.pool1.forward(conv1_out)
        conv2_out = self.relu2.forward(self.conv2.forward(pool1_out))
        pool2_out = self.pool2.forward(conv2_out)
        fc1_out = self.fc1.forward(pool2_out)
        fc2_out = self.fc2.forward(fc1_out)
        fc3_out = self.fc3.forward(fc2_out)
        return fc3_out

    def train(self, batch_size, epochs, lr, images, test_images,labels,test_labels):
        train_loss_record = []
        train_acc_record = []
        val_loss_record = []
        val_acc_record = []

        for epoch in range(epochs):
            learning_rate = lr
            batch_loss = 0
            batch_acc = 0
            val_acc = 0
            val_loss = 0

            # train
            train_acc = 0
            train_loss = 0
            for i in range(int(images.shape[0] / batch_size)):
                img = images[i * batch_size:(i + 1) * batch_size].reshape((batch_size, 28, 28, 1))
                img = np.pad(img, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')  #
                label = labels[i * batch_size:(i + 1) * batch_size]

                fc3_out = self.forward(img)
                batch_loss += self.sf.cal_loss(fc3_out, np.array(label))
                train_loss += self.sf.cal_loss(fc3_out, np.array(label))

                for j in range(batch_size):
                    if np.argmax(self.sf.softmax[j]) == label[j]:
                        batch_acc += 1
                        train_acc += 1

                self.sf.gradient()
                self.conv1.gradient(self.relu1.gradient(self.pool1.gradient(
                    self.conv2.gradient(self.relu2.gradient(self.pool2.gradient(
                        self.fc1.gradient(self.fc2.gradient(self.fc3.gradient(self.sf.eta)))))))))

                if i % 1 == 0:
                    self.fc3.backward(alpha=learning_rate, weight_decay=0.0004)
                    self.fc2.backward(alpha=learning_rate, weight_decay=0.0004)
                    self.fc1.backward(alpha=learning_rate, weight_decay=0.0004)
                    self.conv2.backward(alpha=learning_rate, weight_decay=0.0004)
                    self.conv1.backward(alpha=learning_rate, weight_decay=0.0004)

                    if i % 100 == 0:
                        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + \
                              "  epoch: %d ,  batch: %5d , avg_batch_acc: %.4f  avg_batch_loss: %.4f  learning_rate %f" % (
                              epoch,
                              i, batch_acc / float(
                                  batch_size), batch_loss / batch_size, learning_rate))

                    batch_loss = 0
                    batch_acc = 0

            print(time.strftime("%Y-%m-%d %H:%M:%S",
                                time.localtime()) + "  epoch: %d , train_acc: %.4f  avg_train_loss: %.4f" % (
                      epoch, train_acc / float(images.shape[0]), train_loss / images.shape[0]))

            train_loss_record.append(train_loss / images.shape[0])
            train_acc_record.append(train_acc / float(images.shape[0]))

            # validation
            for i in range(test_images.shape[0] // batch_size):
                img = test_images[i * batch_size:(i + 1) * batch_size].reshape([batch_size, 28, 28, 1])
                img = np.pad(img, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')  #
                label = test_labels[i * batch_size:(i + 1) * batch_size]

                fc3_out = self.forward(img)
                val_loss += self.sf.cal_loss(fc3_out, np.array(label))

                for j in range(batch_size):
                    if np.argmax(self.sf.softmax[j]) == label[j]:
                        val_acc += 1

            print(time.strftime("%Y-%m-%d %H:%M:%S",
                                time.localtime()) + "  epoch: %d , val_acc: %.4f  avg_val_loss: %.4f" % (
                      epoch, val_acc / float(test_images.shape[0]), val_loss / test_images.shape[0]))

            val_acc_record.append(val_acc / float(test_images.shape[0]))
            val_loss_record.append(val_loss / test_images.shape[0])

        iters = range(len(train_acc_record))
        plt.figure()
        # acc
        plt.plot(iters, train_acc_record, 'r', label='train acc')
        # loss
        plt.plot(iters, train_loss_record, 'g', label='train loss')
        # val_acc
        plt.plot(iters, val_acc_record, 'b', label='val acc')
        # val_loss
        plt.plot(iters, val_loss_record, 'k', label='val loss')
        plt.grid(True)
        plt.xlabel('epochs')
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.savefig("plot_result/numpylr{}epochs{}bs{}.png".format(lr, epochs, batch_size))
        plt.show()

    def test(self, img):#1*32*32*1
        img = img.repeat(self.batch_size, axis=0)
        fc3_out = self.forward(img)
        pre = self.sf.predict(fc3_out)
        result = np.argmax(self.sf.softmax[0])
        return result

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model",
                    help="path to output model after training")
    args = vars(ap.parse_args())
    batch_size = 64
    epochs = 3
    lr = 1e-4

    # grab the MNIST dataset
    print("[INFO] accessing MNIST...")
    ((images, labels), (test_images, test_labels)) = mnist.load_data()

    model = LeNet(batch_size)
    model.train(batch_size, epochs, lr, images=images, test_images=test_images, labels=labels, test_labels=test_labels)

    print("[INFO] serializing digit model...")
    output_hal = open("output/numpy_LeNet{}{}{}.pkl".format(batch_size, lr, epochs), 'wb')
    str = pickle.dumps(model)
    output_hal.write(str)
    output_hal.close()








