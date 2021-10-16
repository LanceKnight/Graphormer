import math
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, auc, roc_curve


def calculate_logAUC(true_y, predicted_score, FPR_range=None):

    if FPR_range is not None:
        range1 = FPR_range[0]
        range2 = FPR_range[1]
        if (range1 >= range2):
            raise Exception('FPR range2 must be greater than range1')
    # print(f'true_y:{true_y}, predicted_score:{predicted_score}')
    fpr, tpr, thresholds = roc_curve(true_y, predicted_score, pos_label=1)
    x = fpr
    y = tpr
    x = np.log(x)

    y1 = np.append(y, np.interp(range1, x, y))
    y = np.append(y1, np.interp(range2, x, y))
    x = np.append(x, range1)
    x = np.append(x, range2)

    x = np.sort(x)
    y = np.sort(y)

    range1_idx = np.where(x == range1)[-1][-1]
    range2_idx = np.where(x == range2)[-1][-1]

    trim_x = x[range1_idx:range2_idx + 1]
    trim_y = y[range1_idx:range2_idx + 1]

    area = auc(trim_x, trim_y)
    return area