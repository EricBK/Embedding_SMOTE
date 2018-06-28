"""
将 FM生成的 dense feature扩充加入原有的feature，看是否有增益
"""
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from Configure import *
import pandas as pd
args = parser.parse_args()

train_original = pd.read_csv(args.train_original)
test_original = pd.read_csv(args.test_original)

train_dense = pd.read_csv(args.train_dense)
test_dense = pd.read_csv(args.test_dense)

del train_dense['label']
del test_dense['label']
train_merge = pd.merge(left=train_original,right=train_dense,on='id')
test_merge = pd.merge(left=test_original,right=test_dense,on='id')

train_merge.to_csv(args.train_merge, index=False)
test_merge.to_csv(args.test_merge, index=False)

