import pandas as pd
import numpy as np
import scipy.stats
import os
import pickle
from sklearn.metrics import roc_auc_score
import ast

# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def delong_roc_variance(ground_truth, predictions):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes log(p-value) for hypothesis that two ROC AUCs are different
    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model,
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov)


def main_delong(labels, prob1, prob2):

    assert len(prob1) == len(prob2)
    log_p = delong_roc_test(labels, prob1, prob2).item()
    p_val = 10**log_p

    return p_val


if __name__ == '__main__':
    # directory path with csv file that contains the details
    detail_csv_path = '/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/chris/CT-CLIP/experiment_results/linear_probing/group_by_dataset_modelsOfInterest_Details'
    
    # directory path with csv file that DONOT contains the details
    abbreviated_csv_path = '/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/chris/CT-CLIP/experiment_results/linear_probing/group_by_dataset_modelsOfInterest'

    # anchors: the model that you want to compare other baseline for delong signficiant test
    anchors = ["swin_exp_infoNCE", "resnet_exp_infoNCE", "swin_pretrained_exp_infoNCE", "resnet_pretrained_exp_infoNCE", "bi-mamba"]

    all_csvs = [_file for _file in os.listdir(detail_csv_path) if '.csv' in _file]
    for detail_csv_file in all_csvs:
        # get the corresponding csv file from the clean directory
        abbreviated_csv_file = os.path.join(abbreviated_csv_path, os.path.basename(detail_csv_file))

        # get the matching dataframe of the same csv file
        detail_df = pd.read_csv(os.path.join(detail_csv_path, detail_csv_file))

        # Convert pred_probs column (string of list) to actual list
        detail_df["pred_probs"] = detail_df["pred_probs"].apply(ast.literal_eval)
        detail_df["labels"] = detail_df["labels"].apply(ast.literal_eval)

        # Identify models containing anchor substrings (Group A)
        anchor_mask = detail_df["model"].astype(str).apply(lambda x: x in anchors)
        anchor_df = detail_df[anchor_mask]  # Models that are in the anchors

        # Identify models that are **not** in the anchors list (Group B)
        non_anchor_df = detail_df[~anchor_mask]  # Models that do not contain anchor substrings

        # List to store result tuples (list_A, list_B)
        pairs_list = []

        # Iterate over each row in anchor models and pair with non-anchor models
        for a_index, row_A in anchor_df.iterrows():
            if row_A['model'] == 'bi-mamba': # in bi-mamba, the label list is different than the ones ran in the cluster.
                continue

            pred_probs_A = row_A["pred_probs"]  # Now it's a list
            model_A_index = row_A['model']
            for b_index, row_B in non_anchor_df.iterrows():
                pred_probs_B = row_B["pred_probs"]  # Now it's a list
                model_B_index = row_B['model']
                # Ensure the labels match
                if not np.array_equal(row_A['labels'], row_B['labels']):
                    raise ValueError("Error: The label lists from both pickle files do not match!")
                pairs_list.append((row_A['labels'], (a_index, pred_probs_A), (b_index, pred_probs_B)))
        print('size of the combinations: ', len(pairs_list))

        # for each of the pair, compute the delong test.
        abbr_df = pd.read_csv(abbreviated_csv_file)
        # abbr_df['delong_stats'] = [None for _ in range(abbr_df.shape[0])]
        pvals = ['None' for _ in range(abbr_df.shape[0])]
        for pair in pairs_list:
            labels = pair[0]
            model_A_index, probs_A = pair[1]
            model_B_index, probs_B = pair[2]
            p_val = main_delong(np.array(labels), np.array(probs_A), np.array(probs_B))
            if not isinstance(pvals[model_A_index], list) :
                pvals[model_A_index] = []
            pvals[model_A_index].append((abbr_df.iloc[model_B_index]['model'], p_val))
        abbr_df['delong_stats'] = pvals
        abbr_df.to_csv(abbreviated_csv_file)