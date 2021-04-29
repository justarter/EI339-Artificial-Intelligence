from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import cv2
from math import *
import numpy as np

class LossHistory(Callback):  # 记录训练时数据并绘图
    '''
    在模型开始的时候定义四个属性，每一个属性都是字典类型，存储相对应的值和epoch
    '''
    def __init__(self, lr, epochs, bs):
        Callback.__init__(self)
        self.lr = lr
        self.epochs = epochs
        self.bs = bs

    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    # 在每一个batch结束后记录相应的值
    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('accuracy'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_accuracy'))

    # 在每一个epoch之后记录相应的值
    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('accuracy'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_accuracy'))

    def loss_plot(self, loss_type):
        '''
        loss_type：指的是 'epoch'或者是'batch'，分别表示是一个batch之后记录还是一个epoch之后记录
        '''
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.savefig("plot_result/keraslr{}epochs{}bs{}.png".format(self.lr, self.epochs, self.bs))
        plt.show()

def rotate(img, degree):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2

    rotate_img = cv2.warpAffine(img, matRotation, (28, 28), borderValue=(0,0,0))
    return rotate_img

def plus_augmentation(trainData_CN, trainLabels_CN):
    degrees = [5,3,-3, -5]
    rotate_train = [[] for _ in range(len(degrees))]
    trainData_out = trainData_CN.copy()
    trainLabels_out = trainLabels_CN.copy()

    for i in range(len(degrees)):
        trainLabels_out = np.concatenate((trainLabels_out, trainLabels_CN), axis=0)
        degree = degrees[i]
        for j in range(len(trainData_CN)):
            rotate_train[i].append(rotate(trainData_CN[j], degree))
        rotate_train[i] = np.array(rotate_train[i])
        trainData_out = np.concatenate((trainData_out, rotate_train[i]), axis=0)

    return trainData_out, trainLabels_out

'''
def sample(trainData, trainLabels, trainData_CN, trainLabels_CN, BS):
    EN_num = len(trainData)
    CN_num = len(trainData_CN)
    EN_ids = np.random.randint(EN_num, size=BS//2)
    CN_ids = np.random.randint(CN_num, size=BS//2)
    data_BS = []
    label_BS = []
    for i in range(BS//2):
        data_BS.append(trainData[EN_ids[i]])
        label_BS.append(trainLabels[EN_ids[i]])
        data_BS.append(trainData_CN[CN_ids[i]])
        label_BS.append(trainLabels_CN[CN_ids[i]])
    data_BS = np.array(data_BS)
    label_BS = np.array(label_BS)
    return data_BS, label_BS
'''
