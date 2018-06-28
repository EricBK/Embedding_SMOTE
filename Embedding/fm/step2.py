import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append('/'.join(rootPath.split('/')[:-1]))

import numpy as np
import pickle
from Configure import *
args = parser.parse_args()

direct_encoding_fields = ["t{}".format(i+1) for i in range(args.n_estimator)]
fm_sets_dir = args.fm_sets_dir
# load direct encoding fields
with open(os.path.join(fm_sets_dir,'t1.pkl'),'rb') as f:
    t1 = pickle.load(f)

with open(os.path.join(fm_sets_dir,'t2.pkl'),'rb') as f:
    t2 = pickle.load(f)

with open(os.path.join(fm_sets_dir,'t3.pkl'),'rb') as f:
    t3 = pickle.load(f)

with open(os.path.join(fm_sets_dir,'t4.pkl'),'rb') as f:
    t4 = pickle.load(f)

with open(os.path.join(fm_sets_dir,'t5.pkl'),'rb') as f:
    t5 = pickle.load(f)

with open(os.path.join(fm_sets_dir,'t6.pkl'),'rb') as f:
    t6 = pickle.load(f)

with open(os.path.join(fm_sets_dir,'t7.pkl'),'rb') as f:
    t7 = pickle.load(f)

with open(os.path.join(fm_sets_dir,'t8.pkl'),'rb') as f:
    t8 = pickle.load(f)

with open(os.path.join(fm_sets_dir,'t9.pkl'),'rb') as f:
    t9 = pickle.load(f)

with open(os.path.join(fm_sets_dir,'t10.pkl'),'rb') as f:
    t10 = pickle.load(f)


field_dict = {}
feature2field = {}
field_index = 0     # 记录 field的个数
ind = 0             # 记录特征长度

for field in direct_encoding_fields:
    # value to one-hot-encoding index dict
    field_sets = eval(field)
    for value in range(max(list(field_sets))):
        field_dict[value+1] = ind
        feature2field[ind] = field_index
        ind += 1
    field_index += 1
    with open('./Embedding/fm/dicts/'+field+'.pkl', 'wb') as f:
        pickle.dump(field_dict, f)

with open('./Embedding/fm/feature2field.pkl', 'wb') as f:
    pickle.dump(feature2field, f)