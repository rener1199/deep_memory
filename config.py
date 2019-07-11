'''
config file
'''
n_one_hot_slot = 32
n_mul_hot_slot = 5
max_len_per_slot = 10 # max num of fts in one mul-hot slot
num_csv_col = 83 # num of cols in the csv file
batch_size = 64
layer_dim = [512, 256, 64, 1]
wide_one_hot_index =[0,1,2,3,4,6,7,8,10,11,12,13,14]
wide_mul_hot_index = [0,1,2]

train_data_day = ['0225','0226','0227','0228','0301','0302','0303','0304','0305']
val_data_day = ['0306']
test_data_day = ['0307']
predict_data_day = ['0307']

pre = '/path/to/file/pre'
suf = '.csv' # suf = '.csv'
train_file_name = [pre + date + suf for date in train_data_day]
val_file_name = [pre + date + suf for date in val_data_day]
test_file_name = [pre + date + suf for date in test_data_day]
predict_file_name = [pre + date + suf for date in predict_data_day]

n_ft = 20392345
time_style = '%Y-%m-%d %H:%M:%S'
loss_wgt = 1.0
output_file_name = 'deep_memory'
k = 10 # embedding size / number of latent factors
eta = 0.005 # learning rate
opt_alg = 'Adagrad' # 'Adam'
max_num_lower_ct = 100
n_epoch = 1 # number of times to loop over the whole data sets
record_step_size = 4000 # record the loss and auc after xx mini-batches
mem_size = 64
mem_fea_index = 0
