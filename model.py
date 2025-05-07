import xgboost as xgb
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from loss import *
from xgboost.training import train as xgb_train



class FocalXGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, params=None, num_boost_round=100, alpha=0.25, gamma=2.0):
        self.params = params or {
            'max_depth': 3,
            'eta': 0.1,
            'verbosity': 1,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        }
        self.num_boost_round = num_boost_round
        self.alpha = alpha
        self.gamma = gamma
        self.booster = None

    def fit(self, X, y, eval_set=None, **kwargs):
        dtrain = xgb.DMatrix(X, label=y)
        evals = []
        if eval_set:
            for i, (X_eval, y_eval) in enumerate(eval_set):
                evals.append((xgb.DMatrix(X_eval, label=y_eval), f"eval_{i}"))

        self.booster = xgb_train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            feval=focal_loss_metric(alpha=self.alpha, gamma=self.gamma),
            **kwargs
        )
        return self

    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        probs = self.booster.predict(dtest)
        return np.vstack([1 - probs, probs]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)