"""
读入数据，并根据ffm训练好的模型，将生成的矩阵保存下来，并转成 低维稠密向量进行保存
"""

import tensorflow as tf
import os
import sys
import pickle
import numpy as np
import pandas as pd
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from Configure import *
args = parser.parse_args()
model_dir = "./Embedding/fm/checkpoints/"
selected = 54500
ckpt_file1 = "model-{}.meta".format(selected)
ckpt_file2 = "model-{}".format(selected)

train_dataset = args.train_onehot
test_dataset = args.test_onehot
train_dense_file = args.train_dense
test_dense_file = args.test_dense

field_num = 10
embedding_dim = 5
def to_dense(feature_vectors, cur_sample):
    """
    使用 feature_vectors 将 sample转化成dense格式的
    :param feature_vectors:
    :param cur_sample:
    :return:
    """
    dense_feature = []
    for index in cur_sample[:,-1]:
        dense_feature.extend(feature_vectors[index])
    return dense_feature
def trans_data_to_dense(feature_vectors, dataset = 'train'):
    """
    :param feature_vector:  生成的特征矩阵
    :param dataset: 选择将traindataset进行变成dense类型还是testdataset变成dense类型
    :return: 将生成的稠密数据表示保存在本地
    """
    if dataset == 'train':
        filepath = train_dataset
        saved_file = train_dense_file
    elif dataset == 'test':
        filepath = test_dataset
        saved_file = test_dense_file
    with open(filepath, 'rb') as f:
        test_sparse_data_fraction = pickle.load(f)
    # get number of batches
    num_batches = len(test_sparse_data_fraction)
    all_ids = []
    all_true_labels = []
    all_dense_samples = []
    for ibatch in range(num_batches):
        # batch_size data

        batch_ids = test_sparse_data_fraction[ibatch]['id']
        batch_y = test_sparse_data_fraction[ibatch]['label']
        batch_indexes = np.array(test_sparse_data_fraction[ibatch]['indexes'], dtype=np.int64)
        batch_dense_samples = []
        for i in range(len(batch_indexes)//field_num):
            cur_sample = batch_indexes[i*field_num: (i+1)*field_num]
            cur_sample = to_dense(feature_vectors, cur_sample)
            batch_dense_samples.append(cur_sample)
        all_ids.extend(batch_ids)
        all_true_labels.extend(batch_y)
        all_dense_samples.extend(batch_dense_samples)
    columns = ['id','label']
    columns_other = ["f_{}".format(i+1) for i in range(field_num*embedding_dim)]

    df_base = pd.DataFrame(np.asarray([all_ids,all_true_labels]).T,columns=columns)
    df_dense = pd.DataFrame(np.asarray(all_dense_samples)*(10**13),columns=columns_other)
    df_dense['id'] = all_ids
    df_all = pd.merge(left=df_base,right=df_dense,on='id')

    df_all.to_csv(saved_file,index=False)
def data_trans():
    with tf.Session() as sess:
        reader = tf.train.NewCheckpointReader(tf.train.latest_checkpoint(model_dir))
        print("use {}".format(tf.train.latest_checkpoint(model_dir)))
        # reader = tf.train.NewCheckpointReader(os.path.join(model_dir,ckpt_file2))
        all_variables = reader.get_variable_to_shape_map()
        feature_vectors = reader.get_tensor('interaction_layer/v')    # shape = [128,40]
        trans_data_to_dense(feature_vectors, dataset = 'train')
        trans_data_to_dense(feature_vectors, dataset = 'test')

if __name__ == '__main__':
    data_trans()