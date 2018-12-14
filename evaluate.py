"""
    Define metric: use F1 score for each class to evaluate the model performance.
    Written by Yanyan Zhao.
"""

import torch


def accuracy(outputs, labels):
    outputs = torch.argmax(outputs, dim=1)
    #calculate F1 score for different features.
    f_score = []
    for i in range(6):
        f = f1_score(outputs, labels, i)
        f_score.append(f)

    return f_score

def f1_score(outputs, labels, feature):
    gt = (labels == feature)
    gt_sum = torch.sum(gt).item()

    predict = (outputs == feature)
    predict_sum = torch.sum(predict).item()  

    tp = torch.mul(gt, predict)
    tp_sum = torch.sum(tp).item()

    if gt_sum == 0 and predict_sum == 0:
        return 'None'
    elif gt_sum != 0 and predict_sum == 0:
        recall = tp_sum / gt_sum
        return 'recall: ' + str(recall)
    elif gt_sum == 0 and predict_sum != 0:
        precision = tp_sum / predict_sum
        return 'precision: ' + str(precision)
    else:
        if tp_sum != 0:
            precision = tp_sum / predict_sum
            recall = tp_sum / gt_sum
            F1 = 2 * precision * recall / (precision + recall)
            return 'F1: ' + str(F1)


