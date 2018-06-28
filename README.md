# Embedding_SMOTE
对高维稀疏特征先进行Embedding，变成低维稠密，然后使用SMOTE扩充少数类样本

本程序对执行步骤如下：

1. 使用RandomTreesEmbedding对所有的样本进行one-hot编码；
2. 选择一种Embedding方法将one-hot特征变成低维稠密的；
3. 使用SMOTE对少数类进行过采样；
4. 使用SMOTE_promotion_compare 对SMOTE后的样本和原来的样本进行比较；

