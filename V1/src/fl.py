import numpy as np
import random
from random import seed
myseed = 1
random.seed(myseed)
import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

import json
#from ml_metrics import kappa
from sklearn.metrics import cohen_kappa_score

import tensorflow_probability as tfp

###DATA##########################################################
def loader_mnist():
  (x_train, y_train), (x_test, y_test) = load_data('./mnist.npz')
  num_classes = 10
  theshape = [-1, 28, 28, 1]
  print(x_train.shape)
  print(y_train.shape)

  norm_x_train = x_train.astype("float32") / 255 
  norm_x_test = x_test.astype("float32") / 255


  encoded_y_train = to_categorical(y_train, num_classes=num_classes, dtype="float32")
  encoded_y_test = to_categorical(y_test, num_classes=num_classes, dtype="float32")

  X_train = norm_x_train.reshape(theshape) #norm_x_train.reshape(-1, 28, 28, 1)
  Y_train = encoded_y_train
  X_test = norm_x_test.reshape(theshape) #norm_x_test.reshape(-1, 28, 28, 1)
  Y_test = encoded_y_test
  return X_train, Y_train, X_test, Y_test


def divide_data_between(X_train, y_train, n_devices):
  indexes = np.array(list(range(len(X_train))))
  random.shuffle(indexes)
  Xs_train = []
  ys_train = []
  n_elem_per_d = len(X_train) / n_devices
  for i in range(n_devices):
    first = int(i * n_elem_per_d)
    last = int(first + n_elem_per_d)
    local_indexes = indexes[first: last]
    local_X = X_train.take(local_indexes, axis = 0)

    local_Y = y_train.take(local_indexes, axis = 0)
    
    Xs_train.append(local_X)
    ys_train.append(local_Y)
    
  return Xs_train, ys_train
###DATA##########################################################



###MODEL##########################################################
def create_topology_mnist_conv_28_28_1():
  seed(1)
  tf.random.set_seed(1)
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


def test_model(model, X_test, y_test):
  m = tf.keras.metrics.deserialize({"class_name": "CategoricalAccuracy", "config": {"name": "categorical_accuracy", "dtype": "float32"}})
  mse = tf.keras.losses.deserialize({"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}})

  y_pred = model.predict(X_test) 
  loss = mse(y_test, y_pred).numpy()
  m.reset_states()
  m.update_state(y_test, y_pred)
  acc = m.result().numpy()
  #conf_matrix = json.dumps(tf.math.confusion_matrix(labels=tf.argmax(y_test, 1), predictions=tf.argmax(y_pred, 1)).numpy().tolist())
  conf_matrix = tf.math.confusion_matrix(labels=tf.argmax(y_test, 1), predictions=tf.argmax(y_pred, 1)).numpy().tolist()

  ####COHEN KAPPA SCORE
  target_names = range(10)
  true_labels = np.empty(0, "int32")
  predicted = np.empty(0, "int32")

  for i in range(len(conf_matrix)):
    #print(conf_matrix[i])
    temp_true_labels = np.sum(conf_matrix[i]) 
    temp_ok_predicted_labels = conf_matrix[i][i] 
    true_labels = np.append(true_labels, np.full(temp_true_labels, i))
    predicted = np.append(predicted, np.full(temp_ok_predicted_labels, i))
    for k in range(len(conf_matrix[0])):
      if (k != i):
        predicted = np.append(predicted, np.full(conf_matrix[i][k], k))
  ##
  kappa_score_linear = cohen_kappa_score(true_labels, predicted, weights= 'linear')
  kappa_score_quadratic = cohen_kappa_score(true_labels, predicted, weights= 'quadratic')
  ####
  
  return loss, acc, conf_matrix, kappa_score_linear, kappa_score_quadratic
###MODEL##########################################################


def loss(model, x, y, training, loss_object):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  y_ = model(x, training=False)
  return loss_object(y_true=y, y_pred=y_)

def grad(model, inputs, targets, loss_object):
  with tf.GradientTape() as tape:
    #print("zzz loss_object = " + str(loss_object))
    loss_value = loss(model, inputs, targets, True, loss_object)
    print("zzz loss_value = " + str(loss_value))
  return loss_value, tape.gradient(loss_value, model.trainable_variables)


class JSDFederatedLearning:
  def __init__(self, X_train, y_train, X_test, y_test ):
      self.X_train = X_train
      self.y_train = y_train
      self.X_test = X_test
      self.y_test = y_test
      #INITIAL PARAMETERS

      self.n_devices = 4
      self.range_devices = range(self.n_devices)
      self.learning_rate = 0.01
      self.m_batch_size = 8
      self.Xs_train = []
      self.ys_train = []
      self.models = []
      self.optimizers = []
      self.losses = []
      self.global_aggregations_counter = 0
    
      print("X_train.shape = " +  str(self.X_train.shape))
      self.train_len = len(self.X_train)
      self.total_m_batches = int(self.train_len / self.m_batch_size) # 1 epoch
      print("total_m_batches = " + str(self.total_m_batches))

      self.Xs_train, self.ys_train = divide_data_between(self.X_train, self.y_train, self.n_devices)

      self.total_m_batches_local = self.Xs_train[0].shape[0] / self.m_batch_size
      print("Xs_train[0].shape = " +  str(self.Xs_train[0].shape))
      print("Xs_train[1].shape = " +  str(self.Xs_train[1].shape))
      print(self.ys_train)
      print("total_m_batches_local = " + str(self.total_m_batches_local))

      self.initialize_models_optimizers_losses()

      #TO OPTIMIZE
      # threshold
      ### local_steps
      # precission
      ### n_workers


      #to_quantization = [1] * len(models[0].trainable_variables)

      # Values lower than this threshold are converted to zero
      #threshold = 0.1 # Select threshold using max, min, median, avg



      self.max_epochs = 1
      self.threshold_percentage = [0.] * len(self.models[0].trainable_variables) #5. [0. - 50.]

      self.local_steps = 100
      self.quantization_precission = [32] * len(self.models[0].trainable_variables) # [8-32]
      self.selected_workers_per_aggregation = self.n_devices #4
      self.max_local_steps = 99999999
      self.max_global_aggregations = 99999999



  def initialize_models_optimizers_losses(self):
      for i in range(self.n_devices + 1):
        m = create_topology_mnist_conv_28_28_1()
        self.models.append(m)
        m.summary()
        optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        self.optimizers.append(optimizer)
        print(optimizer)
        if i < self.n_devices:
          loss_object = tf.keras.losses.CategoricalCrossentropy()
          self.losses.append(loss_object)
          print(loss_object)




  #def _train_with_weights(self):
  



  # From global to local
  def _update_local_models(self, selected_devices):
        #Send global model to local model
        for i in range(len(self.models[selected_devices[0]].trainable_variables)):
            for device in selected_devices:
              self.models[device].trainable_variables[i].assign(self.models[self.n_devices].trainable_variables[i])       
    
  def _train_local(self, selected_devices, step_in_epoch):
        #Local Train
        for i in selected_devices: #range(n_devices):
          first = int(step_in_epoch * self.m_batch_size)
          last = int(first + self.m_batch_size)
          local_x = self.Xs_train[i][first:last]
          local_y = self.ys_train[i][first:last]
          if local_x.shape[0] == 0: # TODO -> Revisar esto
            print("step_in_epoch = " + str(step_in_epoch))
            break
          
          loss_value, grads = grad(self.models[i], local_x, local_y, self.losses[i])
          self.optimizers[i].apply_gradients(zip(grads, self.models[i].trainable_variables))
          print("loss_value = " + str(loss_value))    


  def _global_aggregation(self, selected_devices):
          for i in range(len(self.models[selected_devices[0]].trainable_variables)):
              temp = []
              for device in selected_devices:
                  if self.threshold_percentage[i] > 0:
                    threshold = tfp.stats.percentile(tf.math.abs(self.models[device].trainable_variables[i]), q=self.threshold_percentage[i])
                  else:
                    threshold = 0
                  
                  threshold_applied =  tf.where( tf.math.greater_equal(tf.math.abs(self.models[device].trainable_variables[i]), threshold), self.models[device].trainable_variables[i] * 1.0, self.models[device].trainable_variables[i] * 0.0)
                  quantized = tf.quantization.quantize_and_dequantize_v2(threshold_applied,
                                                            input_min = 0,
                                                            input_max= 1,
                                                            num_bits = self.quantization_precission[i],
                                                            signed_input = True,
                                                            range_given=False)

                  #temp.append(quantized)#models[device].trainable_variables[i])
                  #temp.append(models[device].trainable_variables[i])
                  temp.append(quantized)
              
            
              temp_tensor = tf.math.add_n(temp)
              temp_tensor = tf.divide(temp_tensor, len(temp))
              #for device in range(self.n_devices + 1):
              #    self.models[device].trainable_variables[i].assign(temp_tensor) 
              self.models[self.n_devices].trainable_variables[i].assign(temp_tensor) #Global Model

          self.global_aggregations_counter += 1


  def _test(self):
    #TESTING
    for i in range(self.n_devices + 1):
      print(test_model(self.models[i], self.X_test, self.y_test))

  def train_and_test(self):
    ## INITIAL WEIGHTS
    #device_weights = []
    #for device in range(n_devices):
    #    layers = []
    #    for layer in range(len(models[device].trainable_variables)):
    #        layers.append(tf.math.multiply(models[device].trainable_variables[layer], 1))
    #    device_weights.append(layers)
    ##

    #LOCAL TRAIN
    total_local_steps = 0
    for epoch in range(self.max_epochs):
      if total_local_steps > self.max_local_steps:
        print("finishing total_local_steps = " + str(total_local_steps))
        break          

      if self.global_aggregations_counter > self.max_global_aggregations:
        print("finishing self.global_aggregations_counter = " + str(self.global_aggregations_counter))
        break    


      step_in_epoch = 0
      while step_in_epoch < self.total_m_batches_local and total_local_steps < self.max_local_steps and self.global_aggregations_counter < self.max_global_aggregations:
        print("STEP = " + str(step_in_epoch))  
        selected_devices = random.sample(self.range_devices, self.selected_workers_per_aggregation)  
        total_selected = len(selected_devices)  
        print("selected_devices = " + str(selected_devices))

        #Send global model to local model
        self._update_local_models(selected_devices)

        
        #Local steps
        start_step_in_epoch = step_in_epoch
        end_step = start_step_in_epoch + self.local_steps
        while step_in_epoch < end_step and step_in_epoch < self.total_m_batches_local: ## Last global aggregation may have less local steps
          self._train_local(selected_devices, step_in_epoch)
          
          step_in_epoch += 1
          total_local_steps += 1

        
  
        #Global Aggregation
        self._global_aggregation(selected_devices)

        print("last global aggregation in epoch = " + str(epoch))
        print("last global aggregation in step_in_epoch = " + str(step_in_epoch))
        print("last global aggregation in total_local_steps = " + str(total_local_steps))




    print("END self.global_aggregations_counter = " + str(self.global_aggregations_counter))

    #####DEBUG TODO
    #Send global model to local model
    #self._update_local_models(self.range_devices)
    ######DEBUG TODO

    #TESTING
    self._test()



X_train, y_train, X_test, y_test = loader_mnist()

JSDFederatedLearning( X_train, y_train, X_test, y_test).train_and_test()





  
