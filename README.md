# Click-Through Rate Prediction with the User Memory Network
If you use this code, please cite the following paper:
* **Wentao Ouyang, Xiuwu Zhang, Shukui Ren, Li Li, Zhaojie Liu, Yanlong Du. 2019. Click-Through Rate Prediction with the User Memory Network. In DLP-KDD. ACM.**
#### TensorFlow (TF) version
1.3.0

#### Abbreviation
ft - feature, slot == field

## Data Preparation
Data is in the "csv" format, where each row contains an instance.\
Assume there are N unique fts. Fts need to be indexed from 1 to N. Use 0 for missing values or for padding.

We categorize fts as i) **one-hot** or **univalent** (e.g., user id, city) and ii) **mul-hot** or **multivalent** (e.g., words in ad title).

csv data format
* \<label\>\<one-hot fts\>\<mul-hot fts\>

We also need to define the max number of features per mul-hot ft slot (through the "max_len_per_slot" parameter) and perform trimming or padding accordingly. Please refer to the following example for more detail.

### Example
1. original fts (ft_name:ft_value)
* label:0, gender:male, age:27, query:apple, title:apple, title:fruit, title:fresh
* label:1, gender:female, age:35, query:shoes, query:winter, title:shoes, title:winter, title:warm, title:sales

2. csv fts (not converted to ft index yet)
* 0, male, 27, apple, 0, 0, apple, fruit, fresh
* 1, female, 35, shoes, winter, 0, shoes, winter, warm

#### Explanation
csv format settings:\
n_one_hot_slot = 2 # num of one-hot ft slots (gender, age)\
n_mul_hot_slot = 2 # num of mul-hot ft slots (query, title)\
max_len_per_slot = 3 # max num of fts per mul-hot ft slot

For the first instance, the mul-hot ft slot "query" contains only 1 ft "apple". We thus pad (max_len_per_slot - 1) zeros, resulting in "apple, 0, 0".\
For the second instance, the mul-hot ft slot "title" contains 4 fts. We thus only keep the first max_len_per_slot fts.
## Source Code
* config.py -- config file
* ctr_funcs.py -- functions
* model_dnn.py -- DNN model for click-through rate prediction
* model_mem.py -- Deep memory network model for click-through rate prediction
* model_wide_deep.py -- Wide & Deep model for click-through rate prediction
* model_mem_wide_deep.py -- Wide & Deep model with user memory network for click-through rate prediction
* train_mem.py -- Train script for deep memory network, and you can train other models by modify this script

## Run the Code
First revise the config file, and then run the code
```bash
nohup python train_mem.py > [output_file_name] 2>&1 &
```
