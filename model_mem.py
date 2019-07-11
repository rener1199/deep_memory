import numpy as np
import tensorflow as tf
import datetime
import ctr_funcs as func
import os
import shutil


###########################################################
class DnnModel():

  # add mask
  def get_masked_one_hot(self, x_input_one_hot):
    data_mask = tf.cast(tf.greater(x_input_one_hot, 0), tf.float32)
    data_mask = tf.expand_dims(data_mask, axis = 2)
    data_mask = tf.tile(data_mask, (1,1,self.k))
    # output: (?, n_one_hot_slot, k)
    data_embed_one_hot = tf.nn.embedding_lookup(self.emb_mat, x_input_one_hot)
    data_embed_one_hot_masked = tf.multiply(data_embed_one_hot, data_mask)
    return data_embed_one_hot_masked

  def get_masked_mul_hot(self, x_input_mul_hot):
    data_mask = tf.cast(tf.greater(x_input_mul_hot, 0), tf.float32)
    data_mask = tf.expand_dims(data_mask, axis = 3)
    data_mask = tf.tile(data_mask, (1,1,1,self.k))
    # output: (?, n_mul_hot_slot, max_len_per_slot, k)
    data_embed_mul_hot = tf.nn.embedding_lookup(self.emb_mat, x_input_mul_hot)
    data_embed_mul_hot_masked = tf.multiply(data_embed_mul_hot, data_mask)
    # move reduce_sum here
    data_embed_mul_hot_masked = tf.reduce_sum(data_embed_mul_hot_masked, 2)
    return data_embed_mul_hot_masked

  # output: (?, n_one_hot_slot + n_mul_hot_slot, k)
  def get_concate_embed(self):
    data_embed_one_hot = self.get_masked_one_hot(self.x_input_one_hot)
    data_embed_mul_hot = self.get_masked_mul_hot(self.x_input_mul_hot)
    #data_embed_mem = tf.nn.embedding_lookup(self.mem_mat, self.x_input_mem)
    data_embed_concat = tf.concat([data_embed_one_hot, data_embed_mul_hot], 1)
    
    return data_embed_concat

  # input: (?, n_slot, k); output: (?, 1)
  # pass var -- you have two sets of vars
  def get_dnn_output(self, data_embed_concat, weight_list, bias_list, \
		   layer_dim, n_one_hot_slot, n_mul_hot_slot):
    # include output layer
    n_layer = len(layer_dim)
    data_embed_dnn = tf.reshape(data_embed_concat, [-1, (n_one_hot_slot + n_mul_hot_slot)*self.k])
    data_embed_mem0 = tf.nn.embedding_lookup(self.mem_mat0, self.x_input_mem)
    data_embed_mem1 = tf.nn.embedding_lookup(self.mem_mat1, self.x_input_mem)
    data_embed_mem0 = tf.reshape(data_embed_mem0, [-1, self.mem_size])
    data_embed_mem1 = tf.reshape(data_embed_mem1, [-1, self.mem_size])
    embed_dnn = tf.concat([data_embed_dnn, data_embed_mem0, data_embed_mem1], 1)
    cur_layer = embed_dnn
    # loop to create DNN struct
    for i in range(0, n_layer):
	# output layer, linear activation
	if i == n_layer - 1:
          h5 = tf.stop_gradient(cur_layer)
          h5_test = cur_layer
          mem_layer = tf.multiply(h5, data_embed_mem1)
          mem_layer = tf.reduce_sum(mem_layer, 1)
	  cur_layer = tf.matmul(cur_layer, weight_list[i]) + bias_list[i]
	else:
	  cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight_list[i]) + bias_list[i], str(i))
    
    y_hat = cur_layer
    return y_hat, mem_layer, h5, h5_test, data_embed_mem0, data_embed_mem1

  def model(self):
    ###########################################################
    x_input = tf.placeholder(tf.int32, shape=[None, self.total_num_ft_col])
    # shape=[None, n_one_hot_slot]
    self.x_input_one_hot = x_input[:, 0:self.idx1]
    x_input_mul_hot = x_input[:, self.idx1:self.idx2]
    # shape=[None, n_mul_hot_slot, max_len_per_slot]
    self.x_input_mul_hot = tf.reshape(x_input_mul_hot, (-1, self.n_mul_hot_slot, self.max_len_per_slot))
    self.x_input_mem = x_input[:, self.mem_fea_index]
    # target vect
    y_target = tf.placeholder(tf.float32, shape=[None, 1])
    # dropout keep prob
    self.keep_prob = tf.placeholder(tf.float32)
    # emb_mat dim add 1 -> for padding (idx = 0)
    with tf.device('/cpu:0'):
      self.emb_mat = tf.Variable(tf.random_normal([self.n_ft + 1, self.k], stddev=0.01))
      self.mem_mat0 = tf.Variable(tf.random_normal([self.n_ft + 1, self.mem_size], stddev=0.01))
      self.mem_mat1 = tf.Variable(tf.random_normal([self.n_ft + 1, self.mem_size], stddev=0.01))
    ################################
    # include output layer
    n_layer = len(self.layer_dim)
    in_dim = (self.n_one_hot_slot + self.n_mul_hot_slot)*self.k + self.mem_size * 2
    weight_list={}
    bias_list={}

    for i in range(0, n_layer):
      out_dim = self.layer_dim[i]
      weight_list[i] = tf.Variable(tf.random_normal(shape=[in_dim, out_dim], \
                        stddev=np.sqrt(2.0/(in_dim+out_dim))))
      bias_list[i] = tf.Variable(tf.constant(0.1, shape=[out_dim]))
      in_dim = self.layer_dim[i]

    ####### DNN ########
    data_embed_concat = self.get_concate_embed()
    y_hat, mem_layer, h5, h5_test, data_embed_mem0, data_embed_mem1 = self.get_dnn_output(\
                  data_embed_concat,weight_list, bias_list, self.layer_dim, \
                  self.n_one_hot_slot, self.n_mul_hot_slot)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=y_target))
    y_hat_stop = tf.stop_gradient(y_hat)
    h_mem = tf.multiply(mem_layer,y_target - tf.nn.sigmoid(y_hat_stop))
    data_embed_mem = tf.multiply(1-tf.tile(y_target, (1, self.mem_size)), data_embed_mem0) +\
                           tf.multiply(tf.tile(y_target, (1, self.mem_size)), data_embed_mem1)
    loss_mem = tf.losses.mean_squared_error(h5,data_embed_mem)
    loss_f = loss + loss_mem
    loss_test = loss
    return loss_f, y_hat, x_input, y_target, self.keep_prob, loss_test

  def __init__(self,n_ft,k,layer_dim,n_one_hot_slot,n_mul_hot_slot,max_len_per_slot, \
               num_csv_col,wide_one_hot_index,wide_mul_hot_index,mem_size,mem_fea_index):
    # embedding sharing
    self.n_ft = n_ft
    self.k = k
    ## dataset
    self.layer_dim = layer_dim
    self.n_one_hot_slot = n_one_hot_slot
    self.n_mul_hot_slot = n_mul_hot_slot
    self.max_len_per_slot = max_len_per_slot
    self.total_num_ft_col = num_csv_col - 1
    ### dataset
    self.idx1 = n_one_hot_slot
    self.idx2 = self.idx1 + n_mul_hot_slot*max_len_per_slot
    self.mem_size = mem_size
    self.mem_fea_index = mem_fea_index
