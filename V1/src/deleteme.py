###################
#
# MODEL
#
###################
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from random import random
from random import seed



def create_topology_mnist_dense_28_28_1():
  seed(1)
  tf.random.set_seed(1)
  model = Sequential()
  model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
  model.add(tf.keras.layers.Dense(32, activation='relu'))
  model.add(tf.keras.layers.Dense(10))  
  return model


def create_topology_mnist_conv_28_28_1():
  seed(1)
  tf.random.set_random_seed(1)
  model = Sequential()

  model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                   activation ='relu', input_shape = (28,28,1)))
  model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                   activation ='relu'))
  model.add(MaxPool2D(pool_size=(2,2)))
  model.add(Dropout(0.25))


  model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                   activation ='relu'))
  model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                   activation ='relu'))
  model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(256, activation = "relu"))
  model.add(Dropout(0.5))
  model.add(Dense(10, activation = "softmax"))
  return model



###testing quantization
mymodel = create_topology_mnist_conv_28_28_1()
mymodel.summary()

print(mymodel.trainable_variables[0].shape)


print(mymodel.trainable_variables[0])
print(tf.quantization.quantize(mymodel.trainable_variables[0], min_range = 1, max_range =2, T = "qint8"))






###############
maxx = tf.math.abs(tf.math.reduce_max(mymodel.trainable_variables[0]))
minn = tf.math.abs(tf.math.reduce_min(mymodel.trainable_variables[0]))
theresult = tf.quantization.quantize_and_dequantize(mymodel.trainable_variables[0],
                                                        #input_min = -0.01,
                                                        #input_max= 0.01,
                                                        input_min = -0.000001,
                                                        input_max= 1,#tf.math.maximum(maxx, minn),#1,                                                       
                                                        num_bits = 32,
                                                        signed_input = False,
                                                        range_given=True)




print(mymodel.trainable_variables[0])
print(theresult)
print(mymodel.trainable_variables[0].shape)


tf.print(mymodel.trainable_variables[0])
