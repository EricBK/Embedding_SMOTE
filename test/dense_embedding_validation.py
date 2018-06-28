"""
对生成的dense_embedding进行正确性的验证
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from Configure import *
args = parser.parse_args()

dense_type = args.dense_type
assert dense_type in ['fm','sae','pca'], 'please choose a correct type'
# 选择使用哪种 Embedding 方式的dense data
if dense_type == 'fm':
    train_dense_file = args.train_dense
    test_dense_file = args.test_dense
elif dense_type == 'sae':
    train_dense_file = args.train_dense_SAE
    test_dense_file = args.test_dense_SAE
elif dense_type == 'pca':
    train_dense_file = args.train_dense_PCA
    test_dense_file = args.test_dense_PCA

train_dense = pd.read_csv(train_dense_file)
test_dense = pd.read_csv(test_dense_file)

target = 'label'
IDcol = 'id'

x_columns = [x for x in train_dense.columns if x not in [target, IDcol]]
X = train_dense[x_columns]
y = train_dense[target]
X_test = test_dense[x_columns]
y_test = test_dense[target]
def randomForest_test():
    print("\n---------------------RandomForest_test({})---------------------".format(dense_type))
    print('training...')

    rf0 = RandomForestClassifier(oob_score=True, random_state=10)
    rf0.fit(X,y)
    print(rf0.oob_score_)


    y_predprob = rf0.predict_proba(X_test)[:,1]
    print("AUC Score (Test): %f" % metrics.roc_auc_score(y_test,y_predprob))
    """
    param_test1 = {'n_estimators': list(range(10, 71, 10))}
    gsearch1 = GridSearchCV(estimator=RandomForestClassifier(min_samples_split=100,
                                                             min_samples_leaf=20, max_depth=8, max_features='sqrt',
                                                             random_state=10),
                            param_grid=param_test1, scoring='roc_auc', cv=5)
    gsearch1.fit(X, y)
    print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)
    """
def LR_test():
    print("\n---------------------Logistic Regression_test({})---------------------".format(dense_type))

    classifier = LogisticRegression()  # 使用类，参数全是默认的
    classifier.fit(X, y)  # 训练数据来学习，不需要返回值
    y_predict = classifier.predict_proba(X_test)[:,-1]
    print("AUC Score (Test): %f" % metrics.roc_auc_score(y_test,y_predict))

if __name__ == '__main__':
    if args.test_model == 'rf':
        randomForest_test()
    elif args.test_model == 'lr':
        LR_test()
    elif args.test_model == 'ALL':
        randomForest_test()
        LR_test()