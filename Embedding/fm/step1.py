import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append('/'.join(rootPath.split('/')[:-1]))

import pandas as pd
import pickle
from Configure import *
args = parser.parse_args()
# one-hot encoding directly
t1 = set()
t2 = set()
t3 = set()
t4 = set()
t5 = set()
t6 = set()
t7 = set()
t8 = set()
t9 = set()
t10 = set()

train = pd.read_csv(args.train_embedding,chunksize=1000)

for data in train:

    t1_v = set(data['t1'].values)
    t1 = t1 | t1_v

    t2_v = set(data['t2'].values)
    t2 = t2 | t2_v

    t3_v = set(data['t3'].values)
    t3 = t3 | t3_v

    t4_v = set(data['t4'].values)
    t4 = t4 | t4_v

    t5_v = set(data['t5'].values)
    t5 = t5 | t5_v

    t6_v = set(data['t6'].values)
    t6 = t6 | t6_v

    t7_v = set(data['t7'].values)
    t7 = t7 | t7_v

    t8_v = set(data['t8'].values)
    t8 = t8 | t8_v

    t9_v = set(data['t9'].values)
    t9 = t9 | t9_v

    t10_v = set(data['t10'].values)
    t10 = t10 | t10_v

# save dictionaries
fm_sets_dir = args.fm_sets_dir
with open(os.path.join(fm_sets_dir,'t1.pkl'),'wb') as f:
    pickle.dump(t1,f)

with open(os.path.join(fm_sets_dir,'t2.pkl'),'wb') as f:
    pickle.dump(t2,f)

with open(os.path.join(fm_sets_dir,'t3.pkl'),'wb') as f:
    pickle.dump(t3,f)

with open(os.path.join(fm_sets_dir,'t4.pkl'),'wb') as f:
    pickle.dump(t4,f)

with open(os.path.join(fm_sets_dir,'t5.pkl'),'wb') as f:
    pickle.dump(t5,f)

with open(os.path.join(fm_sets_dir,'t6.pkl'),'wb') as f:
    pickle.dump(t6,f)

with open(os.path.join(fm_sets_dir,'t7.pkl'),'wb') as f:
    pickle.dump(t7,f)

with open(os.path.join(fm_sets_dir,'t8.pkl'),'wb') as f:
    pickle.dump(t8,f)

with open(os.path.join(fm_sets_dir,'t9.pkl'),'wb') as f:
    pickle.dump(t9,f)

with open(os.path.join(fm_sets_dir,'t10.pkl'),'wb') as f:
    pickle.dump(t10,f)
