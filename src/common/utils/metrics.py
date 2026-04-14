import numpy as np
from easydict import EasyDict
from sklearn.metrics import roc_curve, auc, confusion_matrix
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def find_best_threshold(y_trues, y_preds):
    '''
        This function is utilized to find the threshold corresponding to the best ACER
        Args:
            y_trues (list): the list of the ground-truth labels, which contains the int data
            y_preds (list): the list of the predicted results, which contains the float data
    '''
    print("Finding best threshold...")
    best_thre = 0.5
    best_metrics = None
    candidate_thres = list(np.unique(np.sort(y_preds)))
    for thre in candidate_thres:
        metrics = cal_metrics(y_trues, y_preds, threshold=thre)
        if best_metrics is None:
            best_metrics = metrics
            best_thre = thre
        elif metrics.ACER < best_metrics.ACER:
            best_metrics = metrics
            best_thre = thre
    print(f"Best threshold is {best_thre}")
    return best_thre, best_metrics


def cal_metrics(y_trues, y_preds, threshold=0.5):
    '''
        This function is utilized to calculate the performance of the methods
        Args:
            y_trues (list): the list of the ground-truth labels, which contains the int data
            y_preds (list): the list of the predicted results, which contains the float data
            threshold (float, optional): 
                'best': calculate the best results
                'auto': calculate the results corresponding to the thresholds of EER
                float: calculate the results of the specific thresholds
    '''

    metrics = EasyDict()

    fpr, tpr, thresholds = roc_curve(y_trues, y_preds)
    metrics.AUC = auc(fpr, tpr)

    try:
        # Guard against degenerate ROC curves (e.g., all preds identical → NaN at boundaries)
        interp_tpr = interp1d(fpr, tpr, kind='linear', fill_value='extrapolate')
        metrics.EER = brentq(lambda x: 1. - x - interp_tpr(x), 0., 1.)
        metrics.Thre = float(interp1d(fpr, thresholds)(metrics.EER))
    except ValueError:
        # Fallback: use threshold=0.5 when EER cannot be computed
        metrics.EER = float('nan')
        metrics.Thre = 0.5

    if threshold == 'best':
        _, best_metrics = find_best_threshold(y_trues, y_preds)
        return best_metrics

    elif threshold == 'auto':
        threshold = metrics.Thre

    prediction = (np.array(y_preds) > threshold).astype(int)

    res = confusion_matrix(y_trues, prediction, labels=[0, 1])
    # Guard against single-class cases (matrix may be 1x1 or 1x2)
    if res.shape == (1, 1):
        if y_trues[0] == 0:
            TP, FN = res[0, 0], 0
            FP, TN = 0, 0
        else:
            TP, FN = 0, 0
            FP, TN = 0, res[0, 0]
    elif res.shape == (1, 2):
        TP, FN = res[0, 0], res[0, 1]
        FP, TN = 0, 0
    elif res.shape == (2, 2):
        TP, FN = res[0, 0], res[0, 1]
        FP, TN = res[1, 0], res[1, 1]
    else:
        raise RuntimeError(f"Unexpected confusion_matrix shape: {res.shape}")

    total = TP + TN + FP + FN
    metrics.ACC = (TP + TN) / total if total > 0 else 0.0

    TP_rate = float(TP / (TP + FN)) if (TP + FN) > 0 else 0.0
    TN_rate = float(TN / (TN + FP)) if (TN + FP) > 0 else 0.0

    denom_apcer = TN + FP
    denom_bpcer = FN + TP
    metrics.APCER = float(FP / denom_apcer) if denom_apcer > 0 else 0.0
    metrics.BPCER = float(FN / denom_bpcer) if denom_bpcer > 0 else 0.0
    metrics.ACER = (metrics.APCER + metrics.BPCER) / 2

    return metrics
