"""
使用去噪自编码对 one-hot类型的特征进行Embedding
"""
import pandas as pd
import numpy as np
import pickle
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append('/'.join(rootPath.split('/')[:-1]))
from Configure import parser
from Embedding.Auto_encoder.deepautoencoder.stacked_autoencoder import *
args = parser.parse_args()

field_num = args.n_estimator

# onehot file 都是 pickle 文件
train_onehot_pkl = args.train_onehot
test_onehot_pkl = args.test_onehot

train_onehot_csv = args.train_onehot_csv
test_onehot_csv = args.test_onehot_csv
generate_onehot_csv = args.generate_onehot_csv

with open('./parameters.conf', 'r') as file:
    parameters = file.readline()
    feature_len = int(parameters.split(':')[1])
def get_onehot(indexes):
    """
    根据每个sample的index值，返回其 onehot特征
    :param indexes:
    :return:
    """
    onehot = [0 for _ in range(feature_len)]
    for i,index in enumerate(indexes[:,1]):
        onehot[index] = 1
    return onehot
def read_from(filepath):
    """
    读入 pickle 文件
    :param filepath:
    :return: one-hot类型的文件
    """
    with open(filepath,'rb') as file:
        sparse_data_fraction = pickle.load(file)
    num_batches = len(sparse_data_fraction)
    all_labels = []
    all_data_onehot = []
    for ibatch in range(num_batches):
        # batch_size data
        batch_y = sparse_data_fraction[ibatch]['label']
        all_labels.extend(batch_y)
        batch_indexes = np.array(sparse_data_fraction[ibatch]['indexes'], dtype=np.int64)

        for i_sample in range(len(batch_indexes)//field_num):
            i_sample_indexes = batch_indexes[i_sample*field_num:(i_sample+1)*field_num]
            i_sample_onehot = get_onehot(i_sample_indexes)
            all_data_onehot.append(i_sample_onehot)
    return np.hstack((np.asarray(all_labels).reshape(-1,1), np.asarray(all_data_onehot)))

if generate_onehot_csv:
    print('saving onehot dataset...')
    train_onehot = read_from(train_onehot_pkl)
    test_onehot  = read_from(test_onehot_pkl)
    columns = ['f_{}'.format(i+1) for i in range(feature_len)]
    pd.DataFrame(data=train_onehot,columns=['label']+columns).to_csv(train_onehot_csv,index_label='id')
    pd.DataFrame(data=test_onehot,columns=['label']+columns).to_csv(test_onehot_csv,index_label='id')
else:
    print("loading onehot dataset...")
    train_onehot_all = pd.read_csv(train_onehot_csv)
    test_onehot_all = pd.read_csv(test_onehot_csv)
    columns = [x for x in train_onehot_all.columns if x not in ['id','label']]
    train_onehot_data = train_onehot_all[columns].values
    test_onehot_data = test_onehot_all[columns].values


model = StackedAutoEncoder(dims=[200, 200], activations=['linear', 'linear'], epoch=[
    3000, 3000], loss='rmse', lr=0.007, batch_size=100, print_step=200,use_latest_para=True)
# 使用 train_onehot 对模型进行训练，并保存模型
model.fit(train_onehot_data)

print('transform and save train&test data (after Stacked Auto Encoder)...')
train_dense_SAE = model.transform(train_onehot_data)
row, col = train_dense_SAE.shape
SAE_columns = ["f_{}".format(i+1) for i in range(col)]
pd.DataFrame(data=np.hstack((train_onehot_all['label'].values.reshape(-1,1),train_dense_SAE)),columns=['label']+SAE_columns).to_csv(
    args.train_dense_SAE, index_label='id'
)
test_dense_SAE = model.transform(test_onehot_data)
pd.DataFrame(data=np.hstack((test_onehot_all['label'].values.reshape(-1,1),test_dense_SAE)),columns=['label']+SAE_columns).to_csv(
    args.test_dense_SAE, index_label='id'
)

# 将SAE编码之后的特征保存到文件中
