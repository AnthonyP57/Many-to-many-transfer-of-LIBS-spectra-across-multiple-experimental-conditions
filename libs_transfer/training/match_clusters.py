import numpy as np
from scipy.optimize import linear_sum_assignment

def map_labels(true_labels, predicted_labels):
    true_labels = np.asarray(true_labels, dtype=int).flatten()
    predicted_labels = np.asarray(predicted_labels, dtype=int).flatten()
    
    D = max(true_labels.max(), predicted_labels.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    
    for i in range(predicted_labels.size):
        w[predicted_labels[i], true_labels[i]] += 1
        
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    
    mapping = dict(zip(row_ind, col_ind))
    
    aligned_predicted = np.array([mapping.get(label, label) for label in predicted_labels])
    
    return aligned_predicted
