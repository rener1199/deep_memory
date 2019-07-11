import numpy as np
import tensorflow as tf
import datetime
import ctr_funcs as func
import config as cfg
import os
import shutil
from tensorflow.python.client import timeline
from model_mem import DnnModel
from dataset import get_dataset_op

# config
str_txt = cfg.output_file_name
base_path = './tmp'
model_saving_addr = base_path + '/deep_memory_' + str_txt + '/'
output_file_name = base_path + '/deep_memory_' + str_txt + '.txt'

# embedding sharing
n_ft = cfg.n_ft
k = cfg.k

## dataset
train_file_name = cfg.train_file_name
val_file_name = cfg.val_file_name
test_file_name = cfg.test_file_name
batch_size = cfg.batch_size
layer_dim = cfg.layer_dim
n_one_hot_slot = cfg.n_one_hot_slot
n_mul_hot_slot = cfg.n_mul_hot_slot
max_len_per_slot = cfg.max_len_per_slot
label_col_idx = 0
num_csv_col = cfg.num_csv_col
record_defaults = [[0]]*num_csv_col
record_defaults[0] = [0.0]
total_num_ft_col = num_csv_col - 1
wide_one_hot_index = cfg.wide_one_hot_index
wide_mul_hot_index = cfg.wide_mul_hot_index

#config memory
mem_size = cfg.mem_size
mem_fea_index = cfg.mem_fea_index

## config of opt alg
eta = cfg.eta
n_epoch = cfg.n_epoch
max_num_lower_ct = cfg.max_num_lower_ct
record_step_size = cfg.record_step_size
opt_alg = cfg.opt_alg

# create dir
if not os.path.exists(base_path):
  os.mkdir(base_path)

### dataset
idx1 = n_one_hot_slot
idx2 = idx1 + n_mul_hot_slot*max_len_per_slot

###########################################################
###########################################################

print('Loading data start!')
tf.set_random_seed(123)

features, labels, training_init_op, validation_init_op, test_init_op = get_dataset_op(\
train_file_name,val_file_name, test_file_name, batch_size, n_epoch)

# prediction
#############################
dnnModel = DnnModel(n_ft,k,layer_dim,n_one_hot_slot,n_mul_hot_slot,max_len_per_slot, \
                   num_csv_col,wide_one_hot_index,wide_mul_hot_index,mem_size,mem_fea_index)
loss, y_hat, x_input, y_target, keep_prob, loss_test = dnnModel.model()
pred_score = tf.sigmoid(y_hat)

if opt_alg == 'Adam':
  optimizer = tf.train.AdamOptimizer(eta).minimize(loss)
else:
  # default
  optimizer = tf.train.AdagradOptimizer(eta).minimize(loss)

########################################
# Launch the graph.
config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7

with tf.Session(config=config) as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
  # Add ops to save and restore all the variables
  saver = tf.train.Saver()    
  train_loss_list = []
  val_loss_list = []
  val_avg_auc_list = []
  epoch_list = []
  best_n_round = 0
  best_val_avg_auc = 0
  early_stop_flag = 0
  lower_ct = 0   

  func.print_time()
  print('Start train loop')
  
  epoch = -1
  for index in range(len(training_init_op)):
    train_data_name = train_file_name[index].split('/')[-1].split('.')[0]
    sess.run(training_init_op[index])
    while True:
      try:           
        epoch += 1
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        train_ft_inst, train_label_inst = sess.run([features, labels])
        train_label_inst = np.transpose([train_label_inst])   
        #train                
        sess.run(optimizer, feed_dict={x_input:train_ft_inst, \
                                           y_target:train_label_inst, \
                                           keep_prob:0.5}, \
                                           options=options, run_metadata=run_metadata)
      except tf.errors.OutOfRangeError:
        print('Train %s finish!' % (train_data_name))
        break
    
    sess.run(validation_init_op)
    val_pred_score_all = []
    val_label_all = []
    val_loss_all = []
    while True:
      try:
        cur_val_ft, cur_val_label = sess.run([features, labels])
        cur_val_label_inst = np.transpose([cur_val_label])
        # pred score
        cur_val_pred_score, cur_val_loss = sess.run([pred_score, loss_test], feed_dict={ \
                                            x_input:cur_val_ft, \
                                            y_target:cur_val_label_inst, \
                                            keep_prob:1.0})
        val_pred_score_all.append(cur_val_pred_score.flatten())
        val_label_all.append(cur_val_label)
        val_loss_all.append(cur_val_loss)
      except:
        break
    val_pred_score_re = func.list_flatten(val_pred_score_all)
    val_label_re = func.list_flatten(val_label_all)
    val_auc_temp, _, _ = func.cal_auc(val_pred_score_re, val_label_re)
    val_loss_mean = np.mean(np.array(val_loss_all))
    val_avg_auc_list.append(val_auc_temp)
    val_loss_list.append(val_loss_mean)
    
    auc_and_loss = [ val_auc_temp, val_loss_mean]
    auc_and_loss = [np.round(xx,4) for xx in auc_and_loss]
    func.print_time()
    print('Generation epoch %s. Val Avg AUC: {:.4f}. Val Loss :{:.4f}.'\
                      .format(*auc_and_loss) % train_data_name)
    if val_auc_temp > best_val_avg_auc:
      best_val_avg_auc = val_auc_temp
      best_n_round = epoch
      # Save the variables to disk
      save_path = saver.save(sess, model_saving_addr)
      print("Model saved in file: %s" % save_path)

  # after training
  saver.restore(sess, model_saving_addr)
  print("Model restored.")
   
  # load test data
  test_pred_score_all = []
  test_label_all = []
  test_loss_all = []
  sess.run(test_init_op)
  try:
    while True:
      test_ft_inst, test_label_inst = sess.run([features, labels])
      cur_test_pred_score = sess.run(pred_score, feed_dict={ \
                              x_input:test_ft_inst, keep_prob:1.0})
      test_pred_score_all.append(cur_test_pred_score.flatten())
      test_label_all.append(test_label_inst)
      
      cur_test_loss = sess.run(loss_test, feed_dict={ \
                              x_input:test_ft_inst, \
                              y_target: np.transpose([test_label_inst]), keep_prob:1.0})
      test_loss_all.append(cur_test_loss)

  except tf.errors.OutOfRangeError:
    print('Done loading testing data 2 -- epoch limit reached')
  
  test_pred_score_re = func.list_flatten(test_pred_score_all)
  test_label_re = func.list_flatten(test_label_all)
  test_auc, _, _ = func.cal_auc(test_pred_score_re, test_label_re)
  test_rmse = func.cal_rmse(test_pred_score_re, test_label_re)
  test_loss = np.mean(test_loss_all)
  
  # rounding
  test_auc = np.round(test_auc, 4)
  test_rmse = np.round(test_rmse, 4)
  test_loss = np.round(test_loss, 5)
  
  train_loss_list = [np.round(xx,4) for xx in train_loss_list]
  val_avg_auc_list = [np.round(xx,4) for xx in val_avg_auc_list]
    
  print('test_auc = ', test_auc)
  print('test_rmse =', test_rmse)
  print('test_loss =', test_loss)    
  print('val_loss_list =', val_loss_list)
  print('val_avg_auc_list =', val_avg_auc_list)
    
  # write output to file
  with open(output_file_name, 'a') as f:
    now = datetime.datetime.now()
    time_str = now.strftime(cfg.time_style)
    f.write(time_str + '\n')
    f.write('learning_rate = ' + str(eta) + ', n_epoch = ' + str(n_epoch) \
            + ', emb_dize = ' + str(k) + '\n')
    f.write('test_auc = ' + str(test_auc) + '\n')
    f.write('test_rmse = ' + str(test_rmse) + '\n')
    f.write('test_loss = ' + str(test_loss) + '\n')
    f.write('val_loss_list =' + str(val_loss_list) + '\n')
    f.write('val_avg_auc_list =' + str(val_avg_auc_list) + '\n')
    f.write('-'*50 + '\n')

