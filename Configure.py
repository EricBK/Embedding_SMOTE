import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
#from Tree_embedding.RandomTreesEmbeddingTest import leaves_num
import argparse
parser = argparse.ArgumentParser()
import pickle
# 记录文件路径
parser.add_argument('--train_original', help='original training data',type=str, default='./data/train_original.csv')
parser.add_argument('--test_original',help='original testing data',type=str, default='./data/test_original.csv')

# 使用RandomTreesEmbedding 对 original 数据进行one-hot编码
parser.add_argument('--train_embedding', help='train embedding file', type=str, default='./data/TrainEmbedding.csv')
parser.add_argument('--test_embedding', help='test embedding file', type=str, default='./data/TestEmbedding.csv')

parser.add_argument('--train_onehot',help='train one hot respresentation',type=str,default='./data/train.pkl')
parser.add_argument('--test_onehot',help='test one hot respresentation',type=str,default='./data/test.pkl')

parser.add_argument('--train_dense',help='train dense representation(fm)',type=str,default='./data/train_dense.csv')
parser.add_argument('--test_dense',help='test dense representation(fm)',type=str,default='./data/test_dense.csv')

# 使用smote 算法之后的文件
parser.add_argument('--train_dense_smote',help='train oversampling(fm)',type=str,default='./data/train_dense_smote.csv')


parser.add_argument('--train_merge',help='merge original and dense features',type=str,default='./data/train_merge.csv')
parser.add_argument('--test_merge',help='merge original and dense features',type=str,default='./data/test_merge.csv')

# Stacked Auto Encoder 里面生成的csv 格式的onehot数据
parser.add_argument('--train_onehot_csv',help='csv file of onehot representation',type=str,default='./data/train_onehot.csv')
parser.add_argument('--test_onehot_csv',help='csv file of onehot representation',type=str,default='./data/test_onehot.csv')

# Stacked Auto Encoder 编码生成的 dense 的特征向量
parser.add_argument('--train_dense_SAE',help='train dataset with SAE embedding',type=str,default='./data/train_dense_SAE.csv')
parser.add_argument('--test_dense_SAE',help='test dataset with SAE embedding', type=str,default='./data/test_dense_SAE.csv')

# PCA 编码生成的 dense 的特征向量
parser.add_argument('--train_dense_PCA',help='train dataset with PCA embedding',type=str,default='./data/train_dense_PCA.csv')
parser.add_argument('--test_dense_PCA',help='test dataset with PCA embedding', type=str,default='./data/test_dense_PCA.csv')

# RandomTreesEmbedding
# 生成的黑样本的比例（1的比例）
parser.add_argument('--pos_rate',help='rate of positive samples',type=float,default=0.01)
parser.add_argument('--n_samples',help='number of all samples',type=int,default=200000)
parser.add_argument('--n_estimator',help='Tree number',type=int,default=10)


# fm
parser.add_argument('--fm_sets_dir',help='fm sets dir',type=str, default='./Embedding/fm/sets')

parser.add_argument('--mode', help='train or test', type=str, default='test')

# parser.add_argument('--feature_length',help='feature length', type=int, default=leaves_num)
parser.add_argument('--epochs',help='training epochs',type=int,default=120)

parser.add_argument('--lr',help='learning rate',type=float, default=0.01)
parser.add_argument('--batch_size',type=int,default=512)
parser.add_argument('--reg_l1',type=float,default=0)
parser.add_argument('--reg_l2',type=float,default=2e-2)
parser.add_argument('--k',help='laten feature length',type=int, default=5)

# dense_embedding_validation
parser.add_argument('--test_model',help='choose a model to check dense embedding validation',type=str,default='ALL') # lr, rf
parser.add_argument('--dense_type',help='choose use which dense data set (FM, SAE, PCA)',type=str,default='fm')

# oversampling
parser.add_argument('--oversampling_rate',help='oversamplint_rate',type=float,default=2)

# compare
parser.add_argument('--compare_mode',help='which compare mode',type=int, default=3)
parser.add_argument('--use_model',help='choose a model to compare promotion',type=str,default='lr')

# Stacked Auto Encoder
parser.add_argument('--generate_onehot_csv',help='do you want to generate onehot csv file?',type=bool,default=False)
parser.add_argument('--use_latest_para',help='do you want to use latest trained model?',type=bool,default=True)

parser.add_argument('--dims',help='hidden unit num',type=list,default=[200,200])
parser.add_argument('--activations',help='activations for each hidden layer',type=list,default=['linear','linear'])
parser.add_argument('--sae_epochs',help='training epoch per hidden layer',type=list,default=[3000,3000])
parser.add_argument('--sae_loss',help='training loss',type=str,default='rmse')
parser.add_argument('--sae_lr',help='sae learning rate',type=float,default=0.007)
parser.add_argument('--sae_batch_size',help='sae training batch_size',type=int,default=512)

# PCA
parser.add_argument('--n_components',help='PCA parameter n_components',type=int, default=50)