"""
使用PCA方法对one-hot编码特征进行降维
"""
import pandas as pd
import numpy as np
import os
import sys
import pandas as pd
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append('/'.join(rootPath.split('/')[:-1]))
from Configure import *
from sklearn.decomposition import PCA

args = parser.parse_args()

train_onehot = pd.read_csv(args.train_onehot_csv)
test_onehot = pd.read_csv(args.test_onehot_csv)

target = 'label'
ID = 'id'
columns = [x for x in train_onehot.columns if x not in [target, ID]]
train_onehot_data = train_onehot[columns].values
train_onehot_labels = train_onehot[target].values
test_onehot_data = test_onehot[columns].values
test_onehot_labels = test_onehot[target].values

n_components = args.n_components
pca = PCA(n_components=n_components)
print("fitting PCA...")
pca.fit(train_onehot_data)
print('transforming...')
train_dense_pca = pca.transform(train_onehot_data)
test_dense_pca = pca.transform(test_onehot_data)

print("saving PCA dense data...")
pca_columns = ["f_{}".format(i+1) for i in range(n_components)]
pd.DataFrame(data=np.hstack((train_onehot_labels.reshape(-1,1),train_dense_pca)),columns=['label']+pca_columns).to_csv(
    args.train_dense_PCA, index_label='id'
)
pd.DataFrame(data=np.hstack((test_onehot_labels.reshape(-1,1),test_dense_pca)),columns=['label']+pca_columns).to_csv(
    args.test_dense_PCA, index_label='id'
)
print("All info:{}  ".format(sum(pca.explained_variance_ratio_)),pca.explained_variance_ratio_)
