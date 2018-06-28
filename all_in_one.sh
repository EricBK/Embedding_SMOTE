# 使用randomTreeEmbedding和SMOTE完成对少数类对过采样


# 首先对原始数据进行RandomTreesEmbedding，将特征变成one-hot类型
python ./Embedding/Tree_embedding/RandomTreesEmbeddingTest.py --pos_rate 0.05 --n_samples 200000

# 使用FM之前分别执行step1，step2 和 utilities
python ./Embedding/fm/step1.py
python ./Embedding/fm/step2.py
python ./Embedding/fm/utilities.py

# 使用FM方法对one-hot类型的变量进行分类，中间生成一个 feature_vectors, 可以用来进行特征转化（变成稠密）
python ./Embedding/fm/fm_tensorflow.py --mode train --epochs 120
# 使用FM训练的模型进行测试，并通过evaluate输出预测效果
python ./Embedding/fm/fm_tensorflow.py --mode test
python ./Embedding/fm/evaluate.py

# 提取fm方法生成的 feature_vectors，并将其作用于之前的one-hot变量，变成稠密特征并存储到 data文件夹中
python ./data_transformation/data_trans.py

# 检验生成的稠密向量的分类性能（使用RandomForest,LR）
python ./test/dense_embedding_validation.py --test_model ALL

# 使用SMOTE方法对其中对少数类进行过采样, 并选择哪个dense 文件进行过采样
python ./SMOTE/over_sampling.py --dense_type fm --oversampling_rate 3

# 验证过采样效果(与不采样的样本进行比较）
python ./test/SMOTE_promotion_compare.py --compare_mode 3 --use_model lr

# 将生成的dense feature和原始特征 merge，并保存下来
python ./data_transformation/feature_extend.py

# 比较merge之后的特征和原来的原始特征相比，是否有增益
