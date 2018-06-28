from sklearn.metrics import roc_curve,auc
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("./Embedding/fm/result_regl1_.csv")
true_labels = data['label']
predictions = data['prediction']
print(true_labels.value_counts())
fpr, tpr, thres = roc_curve(y_true=true_labels, y_score=predictions)
this_auc = auc(fpr, tpr)
print(this_auc)
plt.plot(fpr, tpr)
plt.show()