import tensorflow as tf
import config as cfg

train_file_name = cfg.train_file_name
val_file_name = cfg.val_file_name
test_file_name = cfg.test_file_name
batch_size = cfg.batch_size
n_epoch = cfg.n_epoch
num_csv_col = cfg.num_csv_col

def decode_line(record_line):
    label_col_idx=0
    record_defaults = [[0]]*int(num_csv_col)
    record_defaults[0] = [0.0]
    # Default values, in case of empty columns. Also specifies the type of the decoded result.
    cols = tf.decode_csv(record_line, record_defaults=record_defaults)
    # you can only process the data using tf ops
    label = cols.pop(label_col_idx)
    feature = cols
    # Retrieve a single instance
    return feature, label

def create_dataset(filename, batch_size=32, is_shuffle=False, n_repeats=0):
    """create dataset for train and validation dataset"""
    dataset = tf.data.TextLineDataset(filename)
    if n_repeats > 1:
        dataset = dataset.repeat(n_repeats-1)         # for train
    dataset = dataset.map(decode_line)
    # decode and normalize
    if is_shuffle:
        dataset = dataset.shuffle(10000)            # shuffle
    dataset = dataset.batch(batch_size)
    return dataset

def get_dataset_op(training_filenames, validation_filenames, test_filenames, batch_size, num_epochs):
    # Create different datasets
    training_dataset = []
    for training_file in training_filenames:
      training_dataset.append(create_dataset(training_file, batch_size=batch_size, \
                                  is_shuffle=False, n_repeats=num_epochs)) # train_filename
    validation_dataset = create_dataset(validation_filenames, batch_size=batch_size, \
                                  is_shuffle=False, n_repeats=num_epochs) # val_filename
    test_dataset = create_dataset(test_filenames, batch_size=batch_size, \
                                  is_shuffle=False, n_repeats=num_epochs) # val_filename    

    iterator = tf.data.Iterator.from_structure(test_dataset.output_types,
                                           test_dataset.output_shapes)
    features, labels = iterator.get_next()
    training_init_op = []
    for i in range(len(training_filenames)):
      training_init_op.append(iterator.make_initializer(training_dataset[i]))
    validation_init_op = iterator.make_initializer(validation_dataset)
    test_init_op = iterator.make_initializer(test_dataset)
    
    return features, labels, training_init_op, validation_init_op, test_init_op

