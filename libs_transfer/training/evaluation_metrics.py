import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy import spatial
import itertools

from libs_transfer.training.match_clusters import map_labels


def compute_mean_reference_spectra(all_spectra, conditions, labels, target_labels):
    """Dynamically calculates the mean spectra for every combination of condition and target label."""
    num_conditions = conditions.shape[1]
    mean_spectra_dict = {c: {} for c in range(num_conditions)}
    
    label_indices = np.argmax(labels, axis=1)
    cond_indices = np.argmax(conditions, axis=1)

    for c in range(num_conditions):
        current_cond_idx = np.where(cond_indices == c)[0]
        for label in target_labels:
            current_label_idx = np.where(label_indices == label)[0]
            intersect = np.intersect1d(current_cond_idx, current_label_idx)
            
            if len(intersect) > 0:
                mean_spectra_dict[c][label] = np.mean(all_spectra[intersect], axis=0)
            else:
                mean_spectra_dict[c][label] = np.zeros_like(all_spectra[0])
    return mean_spectra_dict

def build_reference_stack(mean_spectra_dict, num_conditions, target_labels):
    """Stacks the mean spectra to match the permutation evaluations in ACVAE."""
    stacked_list = []
    for src_idx, tgt_idx in itertools.permutations(range(num_conditions), 2):
        for label in target_labels:
            stacked_list.append(mean_spectra_dict[tgt_idx][label])
            
    return np.vstack(stacked_list)

def get_raw_spectra_subsets(all_spectra, conditions, labels, target_labels):
    """Dynamically extracts the raw spectra arrays for every combination of condition and target label."""
    num_conditions = conditions.shape[1]
    spectra_dict = {c: {} for c in range(num_conditions)}
    
    label_indices = np.argmax(labels, axis=1)
    cond_indices = np.argmax(conditions, axis=1)

    for c in range(num_conditions):
        current_cond_idx = np.where(cond_indices == c)[0]
        for label in target_labels:
            current_label_idx = np.where(label_indices == label)[0]
            intersect = np.intersect1d(current_cond_idx, current_label_idx)
            
            if len(intersect) > 0:
                spectra_dict[c][label] = all_spectra[intersect]
            else:
                spectra_dict[c][label] = np.array([]).reshape(0, all_spectra.shape[1])
    return spectra_dict

def build_raw_reference_stack(spectra_dict, num_conditions, target_labels):
    """Concatenates the raw spectra to match the permutation evaluations in ACVAE."""
    stacked_list = []
    for src_idx, tgt_idx in itertools.permutations(range(num_conditions), 2):
        for label in target_labels:
            if len(spectra_dict[tgt_idx][label]) > 0:
                stacked_list.append(spectra_dict[tgt_idx][label])
                
    if stacked_list:
        return np.concatenate(stacked_list, axis=0)
    return np.array([])


class PCAMeanSpectraEvaluator:
    """Calculates cosine similarity for mean spectra in PCA space."""
    def __init__(self, reference_spectra, n_components=15):
        self.pca = PCA(n_components=n_components)
        self.reference_spectra_pca = self.pca.fit_transform(reference_spectra)

    def cosine_similarity(self, pred):
        pred_pca = self.pca.transform(pred)
        cos_sim = [1 - spatial.distance.cosine(p, a) for p, a in zip(pred_pca, self.reference_spectra_pca[:len(pred_pca)])]
        return np.mean(cos_sim)

class PCASpectraEvaluator:
    """Calculates cosine similarity for raw stacked spectra in PCA space."""
    def __init__(self, reference_spectra, n_components=15):
        self.pca = PCA(n_components=n_components)
        self.reference_spectra_pca = self.pca.fit_transform(reference_spectra)

    def cosine_similarity(self, pred):
        pred_pca = self.pca.transform(pred)
        cos_sim = [1 - spatial.distance.cosine(p, a) for p, a in zip(pred_pca, self.reference_spectra_pca[:len(pred_pca)])]
        return np.mean(cos_sim)

class KMeansMeanSpectraEvaluator:
    """Calculates cluster matching accuracy for mean spectra."""
    def __init__(self, reference_spectra, n_components=15, k_clusters=6, random_state=42):
        self.pca = PCA(n_components=n_components)
        self.kmeans = KMeans(n_clusters=k_clusters, random_state=random_state)
        self.reference_spectra_pca = self.pca.fit_transform(reference_spectra)
        self.reference_labels = self.kmeans.fit_predict(self.reference_spectra_pca)

    def calc_kmeans_loss(self, tr_spectra):
        tr_spectra_pca = self.pca.transform(tr_spectra)
        tr_labels = self.kmeans.fit_predict(tr_spectra_pca)
        ref_labels_truncated = self.reference_labels[:len(tr_spectra)]
        
        best_match = map_labels(ref_labels_truncated, tr_labels)
        score = accuracy_score(ref_labels_truncated, best_match)
        return score, 1 - score

class KMeansRawSpectraEvaluator:
    """Calculates cluster matching accuracy for raw stacked spectra."""
    def __init__(self, reference_spectra, n_components=15, k_clusters=6, random_state=42):
        self.pca = PCA(n_components=n_components)
        self.k_clusters = k_clusters
        self.random_state = random_state
        self.reference_spectra_pca = self.pca.fit_transform(reference_spectra)
        
        base_kmeans = KMeans(n_clusters=self.k_clusters, random_state=self.random_state)
        self.reference_labels = base_kmeans.fit_predict(self.reference_spectra_pca)

    def calc_kmeans_loss(self, tr_spectra):
        tr_spectra_pca = self.pca.transform(tr_spectra)
        test_kmeans = KMeans(n_clusters=self.k_clusters, random_state=self.random_state)
        tr_labels = test_kmeans.fit_predict(tr_spectra_pca)

        ref_labels_truncated = self.reference_labels[:len(tr_spectra)]
        best_match = map_labels(ref_labels_truncated, tr_labels)
        score = accuracy_score(ref_labels_truncated, best_match)
        return score, 1 - score