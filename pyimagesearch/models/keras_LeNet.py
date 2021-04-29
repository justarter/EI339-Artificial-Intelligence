# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

class LeNet:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model
		model = Sequential()
		inputShape = (height, width, depth)

		# 32*32*1
		model.add(Conv2D(6, (5, 5), padding="valid", input_shape=inputShape))
		model.add(Activation("relu"))
		# 28*28*6
		model.add(MaxPooling2D(pool_size=(2, 2), padding="valid"))
		# 14*14*6
		model.add(Conv2D(16, (5, 5), padding="valid"))
		model.add(Activation("relu"))
		# 10*10*16
		model.add(MaxPooling2D(pool_size=(2, 2), padding="valid"))
		# 5*5*16
		model.add(Flatten())#400
		model.add(Dense(120))
		model.add(Activation("relu"))
		model.add(Dropout(0.5))

		model.add(Dense(84))
		model.add(Activation("relu"))
		model.add(Dropout(0.5))

		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model
