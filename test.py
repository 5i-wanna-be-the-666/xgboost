import numpy as np
from sklearn.datasets import make_classification
from skfeature.function.information_theoretical_based import MRMR

# 用随机数据生成 100 个样本，20 个特征
X, y = make_classification(n_samples=100, n_features=20, random_state=42)

# 设置 mrmr 只保留 5 个特征
selected = MRMR.mrmr(X, y, n_selected_features=5)

print("选中的特征数量:", len(selected))
print("选中的特征索引:", selected)