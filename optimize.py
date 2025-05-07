import optuna
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

def optimize_xgb_multiobjective(X_train, y_train, X_valid, y_valid, n_trials=20):
    def compute_gmean(y_true, y_pred):
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn + 1e-9)
        specificity = tn / (tn + fp + 1e-9)
        return (sensitivity * specificity) ** 0.5

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.1, 10.0),
            'use_label_encoder': False,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'auc'
        }

        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

        preds = model.predict_proba(X_valid)[:, 1]
        y_pred = (preds >= 0.01).astype(int)

        auc = roc_auc_score(y_valid, preds)
        gmean = compute_gmean(y_valid, y_pred)
        return auc, gmean  # 多目标优化

    # 多目标优化
    study = optuna.create_study(directions=["maximize", "maximize"])
    study.optimize(objective, n_trials=n_trials)

    # 输出 Pareto 前沿结果
    print("\n✅ Pareto 最优解数量:", len(study.best_trials))
    for t in study.best_trials:
        print(f"  AUC={t.values[0]:.4f}, G-Mean={t.values[1]:.4f}, params={t.params}")

    # 选择 Pareto 前沿中 AUC + G-Mean 平均值最大的解
    best_trial = max(
        study.best_trials,
        key=lambda t: (t.values[0] + t.values[1]) / 2
    )
    return best_trial.params

# import pandas as pd
# import pymrmr
# from sklearn.preprocessing import LabelEncoder

# def mrmr_feature_selection(X, y, method='MIQ', n_features=10):
#     """
#     使用 mRMR 方法进行特征选择

#     参数：
#         X: ndarray or DataFrame, 特征数据
#         y: 1D array-like, 标签
#         method: 'MIQ' or 'MID'，mRMR 策略
#         n_features: int, 要选择的特征数量

#     返回：
#         X_new: 筛选后的特征
#         y_new: 原始标签（未变）
#     """
#     # 转换为 DataFrame
#     df = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])

#     # 添加标签列
#     y_encoded = LabelEncoder().fit_transform(y)  # mRMR 要求分类标签
#     df['target'] = y_encoded

#     # 应用 mRMR
#     selected_features = pymrmr.mRMR(df, method, n_features)

#     # 返回筛选后的特征
#     X_new = df[selected_features].values
#     return X_new, y

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from skfeature.function.information_theoretical_based import MRMR
from itertools import combinations

def generate_combined_features(X, methods=['add', 'mul']):
    n_samples, n_features = X.shape
    combined = []
    names = []

    for i, j in combinations(range(n_features), 2):
        if 'add' in methods:
            combined.append((X[:, i] + X[:, j]).reshape(-1, 1))
            names.append(f'add_f{i}_f{j}')
        if 'mul' in methods:
            combined.append((X[:, i] * X[:, j]).reshape(-1, 1))
            names.append(f'mul_f{i}_f{j}')
    return (np.hstack(combined) if combined else np.empty((n_samples, 0))), names

def mrmr_with_combinations(X, y, keep_ratio=0.3, min_features=3, combo_methods=['add', 'mul']):
    """
    组合特征 + mRMR 选择 top K
    """
    y_encoded = LabelEncoder().fit_transform(y)

    original_names = [f'f{i}' for i in range(X.shape[1])]
    X_comb, comb_names = generate_combined_features(X, combo_methods)

    X_all = np.hstack([X, X_comb])
    feature_names = original_names + comb_names

    print("特征个数:", X_all.shape[1])

    k = max(min_features, int(X_all.shape[1] * keep_ratio))
    k = min(k, X_all.shape[1] - 1)  # 防止超出
    print("特征个数(按比例保留之后):", k)

    # mRMR 直接用 numpy，不需要 df
    selected_indices = MRMR.mrmr(X_all, y_encoded, n_selected_features=k)
    selected_indices = selected_indices[:k]
    # 正确使用 numpy 方式进行索引
    X_selected = X_all[:, selected_indices]
    selected_names = [feature_names[i] for i in selected_indices]

    print("✅ 最终 X_selected.shape =", X_selected.shape)

    return X_selected, y, selected_names