import numpy as np
np.random.seed(10)
import sys
import os
import pickle
from collections import Counter
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append('/'.join(rootPath.split('/')[:-1]))
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline
import pandas as pd
from Configure import parser
args = parser.parse_args()
n_estimator = args.n_estimator

X, y = make_classification(n_samples=args.n_samples,weights=[1-args.pos_rate, args.pos_rate])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print('saving original datasets...')
# 将训练数据保存下来
train_samples_num, feature_length = np.asarray(X_train).shape
columns = ["t{}".format(i+1) for i in range(feature_length)]
train_labels = np.asarray(y_train).reshape(train_samples_num,1)
train_data = np.asarray(X_train)
train_all = np.hstack([train_labels,train_data])
pd.DataFrame(data=train_all, columns=['label']+columns).to_csv(args.train_original,index_label='id')
# 将测试数据保存下来
test_samples_num, feature_length = np.asarray(X_test).shape
columns = ["t{}".format(i+1) for i in range(feature_length)]
test_labels = np.asarray(y_test).reshape(test_samples_num,1)
test_data = np.asarray(X_test)
test_all = np.hstack([test_labels,test_data])
pd.DataFrame(data=test_all, columns=['label']+columns).to_csv(args.test_original,index_label='id')

rt = RandomTreesEmbedding(max_depth=3, n_estimators=n_estimator,
    random_state=0)

rt.fit(X_train,y_train)

X_test_embedding = rt.apply(X_test)
X_train_embedding = rt.apply(X_train)

field_num = X_train_embedding.shape[1]
field_id = ["t{}".format(i+1) for i in range(field_num)]

leaves_num = sum([max(X_train_embedding[:,i]) for i in range(field_num)])
print("leaves num: {}".format(leaves_num))
with open('./parameters.conf','w') as file:
    file.write("leaves_num:{}".format(leaves_num))
train_data_path = args.train_embedding
test_data_path = args.test_embedding

print('saving RandomTreesEmbedding datasets...')
# 将训练集和测试集的Tree Embedding 保存下来
X_train_embedding_df = pd.DataFrame(data=X_train_embedding,columns=field_id)
# 增加label列
X_train_embedding_df['label'] = y_train
X_train_embedding_df.to_csv(train_data_path,index_label='id')

# 保存测试集
X_test_embedding_df = pd.DataFrame(data=X_test_embedding, columns=field_id)
X_test_embedding_df['label'] = y_test
X_test_embedding_df.to_csv(test_data_path,index_label='id')

