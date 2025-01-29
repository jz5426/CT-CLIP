import pandas as pd
import numpy as np
import scipy.stats
import os
import pickle
from sklearn.metrics import roc_auc_score

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


def main_delong(pickle_file1: str, pickle_file2: str):

    # Load dictionaries from pickle files
    with open(pickle_file1, "rb") as f:
        dict1 = pickle.load(f)
    
    with open(pickle_file2, "rb") as f:
        dict2 = pickle.load(f)

    # Extract labels and predicted probabilities
    labels1 = np.array(dict1['labels'])
    labels2 = np.array(dict2['labels'])

    # Ensure the labels match
    if not np.array_equal(labels1, labels2):
        raise ValueError("Error: The label lists from both pickle files do not match!")

    # Extract probability scores
    probabilities1 = np.array(dict1['pred_probs'])
    probabilities2 = np.array(dict2['pred_probs'])
    
    assert len(probabilities1) == len(probabilities2)

    auc1 = roc_auc_score(labels1, probabilities1, average='micro', multi_class='ovr')
    auc2 = roc_auc_score(labels2, probabilities2, average='micro', multi_class='ovr')
    print(f'auc for the first file: {auc1}; auc for the second file: {auc2}')

    log_p = delong_roc_test(labels1, probabilities1, probabilities2).item()
    p_val = 10**log_p
    print(f'the p value is {p_val}, {'AUCs are significantly different' if p_val < 0.05 else 'No significant difference'}')

    return auc1, auc2, p_val

# f'modeltype_{model_type}__batchstyle_{batch_style}__bs_{batch_size}__lr_{lr}__wd_{wd}__textcl_{self.text_cl_weight}__ctcl_{self.ct_cl_weight}__pretrained_{pretrained_xray_encoder}'

def full_sweep_evaluation(dirpath, data_portion='1', anchor_file='', results_dest=''):
    """
    given the dirpath that contains all the pikle objects (with labels and probs in it) 

    anchor_file: string the filename that we want to base on for the comparison (the results of our pretrained method, either swin or resnet)
    return:
        - a excel file and a dictionary
    """
    if 'swin' in anchor_file.lower():
        basename = 'swin'
    elif 'resnet' in anchor_file.lower():
        basename = 'resnet'
    else:
        print('not a valid anchor file')
        return None
    
    if 'pretrained_false' in anchor_file.lower():
        basename += '_pretrained_false'
    elif 'pretrained_true' in anchor_file.lower():
        basename += '_pretrained_true'
    else:
        print('not a valid anchor file')
        return None

    train_portion_identifier = f'train_portion_{data_portion}'
    
    # List all pickle files in directory
    pickle_files = [f for f in os.listdir(dirpath) if f.endswith('.pkl') and train_portion_identifier in f]
    
    # Ensure anchor file exists in the directory
    anchor_path = os.path.join(dirpath, anchor_file)
    if not os.path.exists(anchor_path):
        raise FileNotFoundError(f"Anchor file {anchor_file} not found in directory {dirpath}")
    
    results = {}
    
    # Compare each pickle file against the anchor file
    for pickle_file in pickle_files:
        if pickle_file == anchor_file:
            continue  # Skip comparing anchor file to itself
        
        comparison_path = os.path.join(dirpath, pickle_file)
        
        try:
            auc1, auc2, p_val = main_delong(anchor_path, comparison_path)
            results[pickle_file] = {'AUC1': auc1, 'AUC2': auc2, 'p-value': p_val}
        except Exception as e:
            print(f"Error processing {pickle_file}: {e}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame.from_dict(results, orient='index')
    
    # Save results to an Excel file if specified
    if results_dest:
        excel_path = os.path.join(results_dest, f'{basename}_auc_comparison_results_{train_portion_identifier}.xlsx')
        results_df.to_excel(excel_path, index=True)
        print(f"Results saved to {excel_path}")
    
    return results



if __name__ == '__main__':
    # two excel files
    # test the 0.01 (should be no signficiant difference, our model caught up)
    # test the 0.025 (could be signficiantly different or no)
    # test the 0.05 (should be signfiicantly different now)
    # test the 0.10 (should be signfiicantly different now) (not sure)
    # test all the data (should be significant different)

#    file1 = '/Users/maxxyouu/Desktop/delong_test/external_lp_data/delong_stats/modeltype_Swin__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_False_50_epoch_xray_features__train_portion_0.025_data.pkl'
#    file2 = '/Users/maxxyouu/Desktop/delong_test/external_lp_data/delong_stats/swin_medclip_features__train_portion_0.025_data.pkl'
#    main_delong(file1, file2)

    full_sweep_evaluation('remaining pickle objects path', '1', 'anchor file', 'destination')