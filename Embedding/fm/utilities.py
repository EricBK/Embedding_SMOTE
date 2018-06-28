# coding:utf-8
import pandas as pd
import pickle
import logging
from scipy.sparse import coo_matrix
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append('/'.join(rootPath.split('/')[:-1]))
from Configure import *
args = parser.parse_args()

def one_hot_representation(sample, fields_dict, isample):
    """
    One hot presentation for every sample data
    :param fields_dict: fields value to array index
    :param sample: sample data, type of pd.series
    :param isample: sample index
    :return: sample index
    """
    index = []
    for field in fields_dict:
        # get index of array
        field_value = sample[field]
        ind = fields_dict[field][field_value]
        index.append([isample,ind])
    return index

def sparse_data_generate(test_data, fields_dict,dataset= 'train'):
    sparse_data = []
    # batch_index
    ibatch = 0
    for data in test_data:
        ids = []
        labels = []
        indexes = []
        for i in range(len(data)):
            sample = data.iloc[i,:]
            ids.append(sample['id'])
            index = one_hot_representation(sample,fields_dict, i)
            indexes.extend(index)
            label = int(sample['label'])
            labels.append(label)
        sparse_data.append({'indexes':indexes, 'id':ids,'label':labels})
        ibatch += 1
        if ibatch % 200 == 0:
            logging.info('{}-th batch has finished'.format(ibatch))
    if dataset == 'train':
        saved_file = args.train_onehot
    elif dataset == 'test':
        saved_file = args.test_onehot
    with open(saved_file,'wb') as f:
        pickle.dump(sparse_data, f)


# generate batch indexes
if __name__ == '__main__':

    fields = ["t{}".format(i+1) for i in range(args.n_estimator)]

    batch_size = 512
    train = pd.read_csv(args.train_embedding, chunksize=batch_size)
    test = pd.read_csv(args.test_embedding, chunksize=batch_size)
    # loading dicts
    fields_dict = {}
    for field in fields:
        with open('./Embedding/fm/dicts/'+field+'.pkl','rb') as f:
            fields_dict[field] = pickle.load(f)
    print("train sparse data generate")
    sparse_data_generate(train, fields_dict, dataset='train')
    print("test sparse data generate")
    sparse_data_generate(test, fields_dict, dataset = 'test')