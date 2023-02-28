import numpy as np
import random

import tensorflow as tf
import os

#disable warnings
try:
  import tensorflow.python.util.deprecation as deprecation
  deprecation._PRINT_DEPRECATION_WARNINGS = False
except:
  pass
#

from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D


import logging

import json
#from ml_metrics import kappa
from sklearn.metrics import cohen_kappa_score



mypercentile = None
myrandom_seed = None
myquantize_and_dequantize = None


if tf.test.gpu_device_name():
  try:
    if type(tf.contrib) != type(tf): tf.contrib._warning = None
    mypercentile = tf.contrib.distributions.percentile
    config = tf.compat.v1.ConfigProto() #config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.enable_eager_execution() #tf.enable_eager_execution()
    myrandom_seed = tf.compat.v1.random.set_random_seed
    myquantize_and_dequantize = tf.quantization.quantize_and_dequantize
  except:
    import tensorflow_probability as tfp
    mypercentile = tfp.stats.percentile
    myrandom_seed = tf.random.set_seed
    myquantize_and_dequantize = tf.quantization.quantize_and_dequantize_v2
else:
    import tensorflow_probability as tfp
    mypercentile = tfp.stats.percentile
    myrandom_seed = tf.random.set_seed
    myquantize_and_dequantize = tf.quantization.quantize_and_dequantize_v2




###DATA##########################################################
def loader_mnist():
  (x_train, y_train), (x_test, y_test) = load_data('./mnist.npz')
  num_classes = 10
  theshape = [-1, 28, 28, 1]
  #logging.info(x_train.shape)
  #logging.info(y_train.shape)

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
  random_state = random.getstate()
  nprandom_state = np.random.get_state()
  random.seed(1)
  myrandom_seed(1) #tf.compat.v1.random.set_random_seed(1) #tf.random.set_random_seed(1)
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

  random.setstate(random_state)
  np.random.set_state(nprandom_state)
  return model


def create_topology_mnist_dense_28_28_1():
  random_state = random.getstate()
  nprandom_state = np.random.get_state()
  random.seed(1)
  myrandom_seed(1) #tf.compat.v1.random.set_random_seed(1) #tf.random.set_random_seed(1)
  model = Sequential()
  model.add(Flatten(input_shape=(28, 28, 1)))
  model.add(Dense(42, activation='relu'))
  model.add(Dense(10, activation = "softmax"))
  random.setstate(random_state)
  np.random.set_state(nprandom_state)
  return model


def test_model(model, X_test, y_test):
  m = tf.keras.metrics.deserialize({"class_name": "CategoricalAccuracy", "config": {"name": "categorical_accuracy", "dtype": "float32"}})
  mse = tf.keras.losses.MeanSquaredError() #tf.keras.losses.deserialize({"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}})

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
    #logging.info(conf_matrix[i])
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
  
  return acc, loss, conf_matrix, kappa_score_linear, kappa_score_quadratic
###MODEL##########################################################


def loss(model, x, y, training, loss_object):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  y_ = model(x, training=False)
  return loss_object(y_true=y, y_pred=y_)

def grad(model, inputs, targets, loss_object):
  with tf.GradientTape() as tape:
    #logging.info("zzz loss_object = " + str(loss_object))
    loss_value = loss(model, inputs, targets, True, loss_object)
    #logging.info("zzz loss_value = " + str(loss_value))
  return loss_value, tape.gradient(loss_value, model.trainable_variables)


class JSDFederatedLearning:
#  def __init__(self, X_train, y_train, X_test, y_test ):
  def __init__(self, X_train, y_train, X_test, y_test,
learning_rate, m_batch_size, n_slaves, quantization_precission,
threshold_percentage, local_steps, selected_workers_per_aggregation,
max_epochs, max_local_steps, max_global_aggregations, chosen_topology):
      self.chosen_topology = chosen_topology
      self.X_train = X_train
      self.y_train = y_train
      self.X_test = X_test
      self.y_test = y_test
      self.learning_rate = learning_rate
      self.m_batch_size = m_batch_size
      self.n_devices = n_slaves
      self.range_devices = range(self.n_devices)
      self.quantization_precission = quantization_precission
      self.threshold_percentage = threshold_percentage
      self.local_steps = local_steps
      self.selected_workers_per_aggregation = selected_workers_per_aggregation
      self.max_epochs = max_epochs
      self.max_local_steps = max_local_steps
      self.max_global_aggregations = max_global_aggregations
 

      #INITIAL PARAMETERS
      self.Xs_train = []
      self.ys_train = []
      self.models = []
      self.optimizers = []
      self.losses = []
      self.global_aggregations_counter = 0
    
      print("\n")
      logging.info("NEW EVALUATION")
      #logging.info("X_train.shape = " +  str(self.X_train.shape))
      self.train_len = len(self.X_train)
      self.total_m_batches = int(self.train_len / self.m_batch_size) # 1 epoch
      logging.info("total_m_batches = " + str(self.total_m_batches))

      self.Xs_train, self.ys_train = divide_data_between(self.X_train, self.y_train, self.n_devices)

      self.total_m_batches_local = self.Xs_train[0].shape[0] / self.m_batch_size
      #logging.info("Xs_train[0].shape = " +  str(self.Xs_train[0].shape))
      #logging.info("Xs_train[1].shape = " +  str(self.Xs_train[1].shape))
      #logging.info(self.ys_train)
      logging.info("total_m_batches_local = " + str(self.total_m_batches_local))

      self.initialize_models_optimizers_losses(chosen_topology)


      #TODO PESOS EN CADA CAPA
      #Calculating communication effort
      total_parameters_in_model = self.models[0].count_params()
      percentage_params_per_layer = []
      for i in range(len(self.models[0].trainable_variables)):
          percentage_params_per_layer.append(tf.math.reduce_prod(self.models[0].trainable_variables[i].shape).numpy() / total_parameters_in_model)
          

      logging.info("len(self.models[0].trainable_variables) = " + str(len(self.models[0].trainable_variables)))
      logging.info("len(quantization_precission) = " + str(len(quantization_precission)))
      logging.info("len(percentage_params_per_layer) = " + str(len(percentage_params_per_layer)))


      self.comm_n_devices = selected_workers_per_aggregation / n_slaves
      self.comm_thres_q = 0
      for i in range(len(threshold_percentage)):
          self.comm_thres_q += (quantization_precission[i] / 32.0) * ((100.0 - threshold_percentage[i]) / 100.0) * percentage_params_per_layer[i]

      self.comm_steps = 1 / local_steps 
      self.communication_fitness = 0.5 * self.comm_steps * self.comm_n_devices +  0.5 * self.comm_steps * self.comm_n_devices * self.comm_thres_q


      logging.info("self.comm_n_devices = " + str(self.comm_n_devices))
      logging.info("self.comm_thres_q = " + str(self.comm_thres_q))
      logging.info("self.comm_steps = " + str(self.comm_steps))
      logging.info("self.communication_fitness = " + str(self.communication_fitness))
      



  def initialize_models_optimizers_losses(self, chosen_topology):
      for i in range(self.n_devices + 1):
        m = None
        if chosen_topology == "CONV":
          m = create_topology_mnist_conv_28_28_1()
        elif chosen_topology == "DENSE":
          m = create_topology_mnist_dense_28_28_1()
        self.models.append(m)
        #m.summary()
        optimizer = None
        if chosen_topology == "CONV":
          optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        elif chosen_topology == "DENSE":
          optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.optimizers.append(optimizer)
        #logging.info(optimizer)
        if i < self.n_devices:
          loss_object = tf.keras.losses.CategoricalCrossentropy()
          self.losses.append(loss_object)
          #logging.info(loss_object)




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
            logging.info("step_in_epoch = " + str(step_in_epoch))
            break
          
          loss_value, grads = grad(self.models[i], local_x, local_y, self.losses[i])
          self.optimizers[i].apply_gradients(zip(grads, self.models[i].trainable_variables))
          #logging.info("loss_value = " + str(loss_value))    


  def _global_aggregation(self, selected_devices):
          for i in range(len(self.models[selected_devices[0]].trainable_variables)):
              temp = []
              for device in selected_devices:
                  if self.threshold_percentage[i] > 0:
                    threshold = mypercentile(tf.math.abs(self.models[device].trainable_variables[i]), q=self.threshold_percentage[i]) #tf.contrib.distributions.percentile(tf.math.abs(self.models[device].trainable_variables[i]), q=self.threshold_percentage[i]) #tfp.stats.percentile(tf.math.abs(self.models[device].trainable_variables[i]), q=self.threshold_percentage[i])
                  else:
                    threshold = 0
                  
                  threshold_applied =  tf.where( tf.math.greater_equal(tf.math.abs(self.models[device].trainable_variables[i]), threshold), self.models[device].trainable_variables[i] * 1.0, self.models[device].trainable_variables[i] * 0.0)
                  quantized = tf.quantization.quantize_and_dequantize(threshold_applied,
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
    #for i in range(self.n_devices + 1):
    #  logging.info(test_model(self.models[i], self.X_test, self.y_test))
    return self.communication_fitness, test_model(self.models[len(self.models) - 1], self.X_test, self.y_test) 

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
        logging.info("finishing total_local_steps = " + str(total_local_steps))
        break          

      if self.global_aggregations_counter > self.max_global_aggregations:
        logging.info("finishing self.global_aggregations_counter = " + str(self.global_aggregations_counter))
        break    


      step_in_epoch = 0
      while step_in_epoch < self.total_m_batches_local and total_local_steps < self.max_local_steps and self.global_aggregations_counter < self.max_global_aggregations:
        logging.info("STEP = " + str(step_in_epoch))  
        selected_devices = random.sample(self.range_devices, self.selected_workers_per_aggregation)  
        total_selected = len(selected_devices)  
        logging.info("selected_devices = " + str(selected_devices))

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

        logging.info("last global aggregation in epoch = " + str(epoch))
        logging.info("last global aggregation in step_in_epoch = " + str(step_in_epoch))
        logging.info("last global aggregation in total_local_steps = " + str(total_local_steps))




    logging.info("END self.global_aggregations_counter = " + str(self.global_aggregations_counter))

    #####DEBUG TODO
    #Send global model to local model
    #self._update_local_models(self.range_devices)
    ######DEBUG TODO

    #TESTING
    return self._test()




class JSDEvaluate():
  def __init__(self):
      self.X_train, self.y_train, self.X_test, self.y_test = loader_mnist()
      

  def evaluate(self, solution, chosen_topology, nn_seed = 1):
    """
    # solution is organised as
        precision for each layer
        number of slaves to communicate
        number of training steps
        threshold for each layer
    """   
    global random
    random_state = random.getstate()
    nprandom_state = np.random.get_state()
    random.seed(nn_seed)
    myrandom_seed(nn_seed) 
    solution = list(map(round, solution)) #solution = list(map(int, solution))

    n_layers = int((len(solution) - 2) / 2)#12
    logging.info("n_layers = " + str(n_layers))
    logging.info("len(solution) = " + str(len(solution)))
  

    #TO OPTIMIZE
    # threshold
    ### local_steps
    # precission
    ### n_workers
    idx_quantization = 0
    idx_slaves = idx_quantization + n_layers
    idx_steps = idx_slaves + 1
    idx_threshold = idx_steps + 1


    idx = idx_quantization
    quantization_precission = []
    while idx < idx_quantization + n_layers:
      quantization_precission.append(solution[idx])
      idx += 1
    
    n_slaves = 4 #solution[idx_slaves]
    
    local_steps = solution[idx_steps]
  
    idx = idx_threshold
    threshold_percentage = []
    while idx < idx_threshold + n_layers:
      threshold_percentage.append(solution[idx] * 1.0)
      idx += 1


    learning_rate = None
    if chosen_topology == "CONV":
      learning_rate = 0.01
    elif chosen_topology == "DENSE":
      learning_rate = 0.001
    m_batch_size = 8
    #n_slaves = 4
    
    #quantization_precission = [32] * n_layers #[32] * len(self.models[0].trainable_variables) # [8-32]
    #threshold_percentage = [0.] * n_layers #[0.] * len(self.models[0].trainable_variables) #5. [0. - 50.]
    #local_steps = 100
    selected_workers_per_aggregation = solution[idx_slaves] # n_slaves
    #FINALIZATION CRITERIA
    max_epochs = 1
    max_local_steps = 99999999
    max_global_aggregations = 99999999


    myfed = JSDFederatedLearning(self.X_train, self.y_train, self.X_test, self.y_test,
    learning_rate, m_batch_size, n_slaves, quantization_precission,
    threshold_percentage, local_steps, selected_workers_per_aggregation,
    max_epochs, max_local_steps, max_global_aggregations, chosen_topology)

    fitness = myfed.train_and_test()

    random.setstate(random_state)
    np.random.set_state(nprandom_state)

    logging.info("fitness = " + str(fitness)) 
    return fitness[1][0], fitness[0]
    #myfed._test()



  

