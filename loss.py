import numpy as np

import numpy as np

def focal_loss(alpha=0.25, gamma=2.0):
    def fl_obj(y_pred, dtrain):
        y_true = dtrain.get_label()
        # sigmoid
        p = 1.0 / (1.0 + np.exp(-y_pred))
        epsilon = 1e-7
        p = np.clip(p, epsilon, 1. - epsilon)

        # gradient
        grad = alpha * y_true * (1 - p) ** gamma * (gamma * p * np.log(p) + p - 1) + \
               (1 - alpha) * (1 - y_true) * p ** gamma * (-gamma * (1 - p) * np.log(1 - p) + (1 - p))

        # simplified hessian (positive and stable)
        hess = alpha * y_true * (1 - p) ** (gamma - 1) * (
            gamma * (gamma + 1) * p * (1 - p) * np.log(p) + (1 - p) * (1 - 2 * p)
        ) + \
               (1 - alpha) * (1 - y_true) * p ** (gamma - 1) * (
                   gamma * (gamma + 1) * p * (1 - p) * np.log(1 - p) + p * (2 * p - 1)
               )

        # clip hessian to avoid numerical issues
        hess = np.clip(hess, 1e-6, 10.0)
        print("grad mean:", np.mean(grad), "min:", np.min(grad), "max:", np.max(grad))
        print("hess mean:", np.mean(hess), "min:", np.min(hess), "max:", np.max(hess))
        return grad, hess
    return fl_obj

def focal_loss_metric(alpha=0.25, gamma=2.0):
    """
    作为评估指标的 Focal Loss（不影响训练）
    """
    def feval(preds, dtrain):
        y_true = dtrain.get_label()
        p = np.clip(preds, 1e-9, 1 - 1e-9)
        loss = -(
            alpha * y_true * ((1 - p) ** gamma) * np.log(p) +
            (1 - alpha) * (1 - y_true) * (p ** gamma) * np.log(1 - p)
        )
        return 'focal_loss', float(np.mean(loss))
    return feval
