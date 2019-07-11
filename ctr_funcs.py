import tensorflow as tf
import numpy as np
import datetime
from sklearn import metrics

def cal_auc(pred_score, label):
    fpr, tpr, thresholds = metrics.roc_curve(label, pred_score, pos_label=1)
    auc_val = metrics.auc(fpr, tpr)
    return auc_val, fpr, tpr

def cal_rmse(pred_score, label):
    mse = metrics.mean_squared_error(label, pred_score)
    rmse = np.sqrt(mse)
    return rmse

def cal_rectified_rmse(pred_score, label, sample_rate):
    for idx, item in enumerate(pred_score):
        pred_score[idx] = item/(item + (1-item)/sample_rate)
    mse = metrics.mean_squared_error(label, pred_score)
    rmse = np.sqrt(mse)
    return rmse

# only works for 2D list
def list_flatten(input_list):
    output_list = [yy for xx in input_list for yy in xx]
    return output_list


def count_lines(file_name):
    num_lines = sum(1 for line in open(file_name, 'rt'))
    return num_lines

# this func is only for avito data
def tf_read_data(file_name_queue, label_col_idx, record_defaults):
    reader = tf.TextLineReader()
    key, value = reader.read(file_name_queue)
    
    # Default values, in case of empty columns. Also specifies the type of the decoded result.
    cols = tf.decode_csv(value, record_defaults=record_defaults)
    # you can only process the data using tf ops
    label = cols.pop(label_col_idx)
    feature = cols
    # Retrieve a single instance
    return feature, label

time_style = '%Y-%m-%d %H:%M:%S'
def print_time():
    now = datetime.datetime.now()
    time_str = now.strftime(time_style)
    print(time_str)

