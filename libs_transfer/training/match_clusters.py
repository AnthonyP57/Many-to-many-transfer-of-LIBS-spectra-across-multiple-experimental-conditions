import numpy as np
from scipy.optimize import linear_sum_assignment

def map_labels(true_labels, predicted):
    true_classes = np.unique(true_labels)
    pred_classes = np.unique(predicted)
    conf_matrix = np.zeros((len(true_classes), len(pred_classes)), dtype=np.int64)
    for i, t in enumerate(true_classes):
        for j, p in enumerate(pred_classes):
            conf_matrix[i, j] = np.sum((true_labels == t) & (predicted == p))

    cost_matrix = -conf_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = {pred_classes[j]: true_classes[i] for i, j in zip(row_ind, col_ind)}
    aligned_predicted = np.array([mapping[label] for label in predicted])

    return aligned_predicted
