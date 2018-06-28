"""
比较：
1. SMOTE过采样之后与过采样之前相比是否有性能上对提升
2. 使用低维稠密且过采样后的样本训练，与原始数据相比是否有性能上的提升
"""
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from Configure import *
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
args = parser.parse_args()

compare_mode = args.compare_mode    # SMOTE自身对比性能是否有提升 / 与原始数据对比性能是否有提升 / 将dense feature融合，性能是否有提升
dense_type = args.dense_type


target = 'label'
IDcol = 'id'



def compare_oversampling_itself(model='lr'):
    """
    比较过采样与不过采样的训练集相比，是否有性能上的提升
    :return:
    """
    print('\n比较在dense数据下，SMOTE是否有效果上的提升----{}...'.format(model))

    train_dense = pd.read_csv(args.train_dense)
    train_dense_oversampling = pd.read_csv(args.train_dense_oversampling)
    test_dense = pd.read_csv(args.test_dense)

    x_columns = [x for x in train_dense.columns if x not in [target, IDcol]]
    X_train = train_dense[x_columns]
    y_train = train_dense[target]
    X_train_oversampling = train_dense_oversampling[x_columns]
    y_train_oversampling = train_dense_oversampling[target]
    X_test = test_dense[x_columns]
    y_test = test_dense[target]

    """ 使用没有过采样的数据集进行分类 """
    print('No oversampling...')
    if model == 'rf':
        rf0 = RandomForestClassifier(oob_score=True, random_state=10)
        rf0.fit(X_train, y_train)
        print("     oob_score:{}".format(rf0.oob_score_))
    elif model == 'lr':
        rf0 = LogisticRegression()
        rf0.fit(X_train, y_train)
    y_predprob = rf0.predict_proba(X_test)[:, 1]
    print("     {} AUC Score (Test): {}" .format("No oversampling",metrics.roc_auc_score(y_test, y_predprob)))

    """ 使用过采样的数据集进行分类 """
    print('Oversampling...')
    if model == 'rf':
        rf1 = RandomForestClassifier(oob_score=True, random_state=10)
        rf1.fit(X_train_oversampling, y_train_oversampling)
        print("     oob_score:{}".format(rf1.oob_score_))
    elif model == 'lr':
        rf1 = LogisticRegression()
        rf1.fit(X_train_oversampling,y_train_oversampling)
    y_predprob = rf1.predict_proba(X_test)[:, 1]
    print("     {} AUC Score (Test): {}" .format("Oversampling",metrics.roc_auc_score(y_test, y_predprob)))
def compare_oversampling_promotion(model='lr'):
    """
    比较过采样之后与原始的数据集相比，是否有性能上的提升
    :return:
    """
    print('\n比较使用dense数据集，经过SMOTE之后，与原来的数据集合相比，是否有效能上的提升----{}...'.format(model))

    train_original = pd.read_csv(args.train_original)
    test_original = pd.read_csv(args.test_original)

    train_dense_oversampling = pd.read_csv(args.train_dense_oversampling)
    test_dense = pd.read_csv(args.test_dense)

    # 过采样的数据
    x_columns_smote = [x for x in train_dense_oversampling.columns if x not in [target, IDcol]]
    X_train_oversampling = train_dense_oversampling[x_columns_smote]
    y_train_oversampling = train_dense_oversampling[target]
    X_test_dense = test_dense[x_columns_smote]
    y_test_dense = test_dense[target]

    # 原始数据
    x_columns_original = [x for x in train_original.columns if x not in [target, IDcol]]
    X_train_original = train_original[x_columns_original]
    y_train_original = train_original[target]

    X_test_original = test_original[x_columns_original]
    y_test_original = test_original[target]

    """ 使用没有过采样的数据集进行分类 """
    print('No oversampling...')
    if model == 'rf':
        rf0 = RandomForestClassifier(oob_score=True, random_state=10)
        rf0.fit(X_train_original, y_train_original)
        print("     oob_score:{}".format(rf0.oob_score_))
    elif model == 'lr':
        rf0 = LogisticRegression()
        rf0.fit(X_train_original, y_train_original)
    y_predprob = rf0.predict_proba(X_test_original)[:, 1]
    print("     {} AUC Score (Test): {}".format("No oversampling", metrics.roc_auc_score(y_test_original, y_predprob)))

    """ 使用过采样的数据集进行分类 """
    print('Oversampling...')
    if model == 'rf':
        rf1 = RandomForestClassifier(oob_score=True, random_state=10)
        rf1.fit(X_train_oversampling, y_train_oversampling)
        print("     oob_score:{}".format(rf1.oob_score_))
    elif model == 'lr':
        rf1 = LogisticRegression()
        rf1.fit(X_train_oversampling,y_train_oversampling)
    y_predprob = rf1.predict_proba(X_test_dense)[:, 1]
    print("     {} AUC Score (Test): {}".format("Oversampling", metrics.roc_auc_score(y_test_dense, y_predprob)))

def compare_feature_fusion_promotion(model='lr'):
    """
    将 生成的dense特征与原始特征进行融合，看与原来是否有增益
    :param model:  rf or lr
    :return:
    """
    print("\n将 生成的dense特征与原始特征进行融合，看与原来是否有增益{}".format(model))
    train_original = pd.read_csv(args.train_original)
    test_original = pd.read_csv(args.test_original)

    train_merge = pd.read_csv(args.train_merge)
    test_merge = pd.read_csv(args.test_merge)

    # 特征融合的数据
    x_columns_merge = [x for x in train_merge.columns if x not in [target, IDcol]]
    X_train_merge = train_merge[x_columns_merge]
    y_train_merge = train_merge[target]
    X_test_merge = test_merge[x_columns_merge]
    y_test_merge = test_merge[target]

    # 原始数据
    x_columns_original = [x for x in train_original.columns if x not in [target, IDcol]]
    X_train_original = train_original[x_columns_original]
    y_train_original = train_original[target]

    X_test_original = test_original[x_columns_original]
    y_test_original = test_original[target]

    """ 使用没有过采样的数据集进行分类 """
    print('No oversampling...')
    if model == 'rf':
        rf0 = RandomForestClassifier(oob_score=True, random_state=10)
        rf0.fit(X_train_original, y_train_original)
        print("     oob_score:{}".format(rf0.oob_score_))
    elif model == 'lr':
        rf0 = LogisticRegression()
        rf0.fit(X_train_original, y_train_original)
    y_predprob = rf0.predict_proba(X_test_original)[:, 1]
    print("     {} AUC Score (Test): {}".format("No oversampling", metrics.roc_auc_score(y_test_original, y_predprob)))

    """ 使用特征融合的数据集进行分类 """
    print('Oversampling...')
    if model == 'rf':
        rf1 = RandomForestClassifier(oob_score=True, random_state=10)
        rf1.fit(X_train_merge, y_train_merge)
        print("     oob_score:{}".format(rf1.oob_score_))
    elif model == 'lr':
        rf1 = LogisticRegression()
        rf1.fit(X_train_merge, y_train_merge)
    y_predprob = rf1.predict_proba(X_test_merge)[:, 1]
    print("     {} AUC Score (Test): {}".format("Oversampling", metrics.roc_auc_score(y_test_merge, y_predprob)))
if __name__ == '__main__':
    assert compare_mode in [1,2,3, 4]
    model = args.use_model
    if compare_mode == 1:   # 比较使用SMOTE是否有增益
        compare_oversampling_itself(model=model)
    elif compare_mode == 2: # 比较使用SMOTE的数据与原始数据相比是否有增益
        compare_oversampling_promotion(model=model)
    elif compare_mode == 3: # 1和2同时比较
        compare_oversampling_itself(model=model)
        compare_oversampling_promotion(model=model)
    elif compare_mode == 4: # 比较dense特征和原始特征融合是否有增益
        compare_feature_fusion_promotion(model=model)


