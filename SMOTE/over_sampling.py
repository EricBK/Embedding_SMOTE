from imblearn.over_sampling import SMOTE
import pandas as pd
from collections import Counter
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from Configure import *
args = parser.parse_args()

# 选择对哪种类型的dense feature 使用SMOTE
dense_type = args.dense_type

if dense_type == 'fm':
    train_dense_file = args.train_dense
    # 将过采样的样本输出
    test_dense_file = args.test_dense
elif dense_type == 'sae':
    train_dense_file = args.train_dense_SAE
    test_dense_file = args.test_dense_SAE

elif dense_type == 'pca':
    train_dense_file = args.train_dense_PCA
    test_dense_file = args.test_dense_PCA

train_dense_file_smote = args.train_dense_smote
oversampling_rate = args.oversampling_rate   # 将少数类过采样的倍率

train_dense = pd.read_csv(train_dense_file)
test_dense = pd.read_csv(test_dense_file)

target = 'label'
IDcol = 'id'

x_columns = [x for x in train_dense.columns if x not in [target, IDcol]]
X_train = train_dense[x_columns]
y_train = train_dense[target]
X_test = test_dense[x_columns]
y_test = test_dense[target]

# For train
print("Original dataset shape {}".format(Counter(y_train)))
minority_num = Counter(y_train)[1]

resample_num = int(minority_num*oversampling_rate) if Counter(y_train)[0] > int(minority_num*oversampling_rate) else Counter(y_train[0])
sm = SMOTE(random_state=10,ratio={1:resample_num})
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
print("Resampled dataset shape {}".format(Counter(y_train_res)))

# 将过采样的结果保存下来
train_dense_oversampling = pd.DataFrame(data=X_train_res,columns=x_columns)
train_dense_oversampling['label'] = y_train_res
train_dense_oversampling.to_csv(train_dense_file_smote,index_label='id')


