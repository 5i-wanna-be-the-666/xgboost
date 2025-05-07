import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from dataset.load_dat import load_keel_dat
from kmeans_smote import KMeansSMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import numpy as np

def compute_gmean(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        print("Warning: confusion matrix is not binary, cannot compute G-Mean.")
        return 0.0

    TN, FP, FN, TP = cm.ravel()
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    gmean = np.sqrt(sensitivity * specificity)
    return gmean
# **EasyEnsemble 实现**
def easy_ensemble(X, y, n_estimators=10):
    """
    EasyEnsemble: 对多数类进行多次欠采样，生成多个子数据集。
    Args:
        X: 特征矩阵
        y: 标签
        n_estimators: 欲生成的弱分类器数量（欠采样次数）
    Returns:
        samples: List of (X_resampled, y_resampled)
    """
    kmeans_smote = KMeansSMOTE(
    kmeans_args={
        'n_clusters': 100
    },
    smote_args={
        'k_neighbors': 10
    }
    )
    rus = RandomUnderSampler(random_state=42)
    samples = []
    print('开始过采样')
    for _ in range(n_estimators):
        #X_resampled, y_resampled = rus.fit_resample(X, y)
        X_resampled, y_resampled = kmeans_smote.fit_resample(X, y)
        samples.append((X_resampled, y_resampled))
    return samples

# **加载数据集（以 glass1 数据集为例）**
def load_keel(name):
    # 从 UCI 或本地加载 'glass1' 数据集
    # 下载地址: https://sci2s.ugr.es/keel/dataset/data/classification/glass1.zip
    # 假设数据已经保存为 glass1.csv
    data = load_keel_dat("dataset/"+name)  # 替换为实际路径
    print(data)
    X = data.iloc[:, :-1].values  # 特征
    y = data.iloc[:, -1].values   # 标签
    return X, y

# 主流程（K 折）
def main():
    name = "abalone19.dat"
    X, y = load_keel(name)
    print(X)
    print(y)
    y = (y == ' positive').astype(int)  # 统一标签为 0 和 1

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    all_auc = []
    all_gmean = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Fold {fold} ---")
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # EasyEnsemble
        n_estimators = 3
        samples = easy_ensemble(X_train, y_train, n_estimators=n_estimators)

        classifiers = []
        for i, (X_resampled, y_resampled) in enumerate(samples):
            print(f"Training classifier {i+1}/{n_estimators} for fold {fold}")
            model = XGBClassifier(
                objective='binary:logistic',
                scale_pos_weight=1,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42
            )
            model.fit(X_resampled, y_resampled)
            classifiers.append(model)

        # 测试集预测
        ensemble_preds = np.zeros_like(y_test, dtype=float)
        for model in classifiers:
            ensemble_preds += model.predict_proba(X_test)[:, 1]
        ensemble_preds /= n_estimators

        y_pred = (ensemble_preds >= 0.5).astype(int)
        auc = roc_auc_score(y_test, ensemble_preds)
        all_auc.append(auc)

        print(classification_report(y_test, y_pred))
        print(f"AUC-ROC for Fold {fold}: {auc:.4f}")
        gmean = compute_gmean(y_test, y_pred)
        all_gmean.append(gmean)
        print(f"G-Mean for Fold {fold}: {gmean:.4f}")

    # 所有折的平均 AUC
    print(f"\nAverage AUC-ROC over {skf.n_splits} folds: {np.mean(all_auc):.4f}")
    print(f"\nAverage G-Mean over {skf.n_splits} folds: {np.mean(all_gmean):.4f}")

if __name__ == "__main__":
    main()