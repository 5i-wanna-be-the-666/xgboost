# 🧬 不平衡分类增强与集成框架：KMeansSMOTE + EasyEnsemble + XGBoost

本项目专注于处理 **极度不平衡的二分类任务**，结合了多种采样与集成策略，包括：

- ✅ 自定义实现的 **KMeansSMOTE**（聚类驱动的过采样）
- ✅ **EasyEnsemble**：多轮欠采样集成策略
- ✅ 集成多个 **XGBoost/FocalXGBoost** 分类器
- ✅ 支持 **K折交叉验证评估**
- ✅ 支持 **Optuna 自动调参（多目标优化）**
- ✅ 支持 SDV 合成样本增强（可选）

---

## 📦 依赖安装

```bash
pip install -r requirements.txt