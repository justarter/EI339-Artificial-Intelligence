EI339_CN: 中文数据集和一些处理函数

layers: 自己实现的LeNet所需的一些卷积层、池化层等函数

output: 训练保存的数字分类器模型，分别为自己实现的（numpy_LeNet）和Keras_LeNet

plot_result: 训练结果图

pyimagesearch/models: keras_LeNet model

pyimagesearch/sudoku: 用来定位的工具函数

test: 测试的十张图片

solve_sudoku_puzzle.py: 主函数，实现从校正识别求解的所有过程

sudoku_solvers.py: 自己写的数独求解类，用来求解数独

train_keras_LeNet.py: 训练Keras实现的LeNet

train_numpy_LeNet.py: 训练自己实现的LeNet

utils.py: 一些工具函数

使用以下命令可以测试图片：

python solve_sudoku_puzzle.py --model output/keras_LeNet1280.00140.h5 --image test/1-1.jpg
