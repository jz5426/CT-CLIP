import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import auc, roc_curve
from sklearn.metrics import precision_recall_curve
import seaborn as sns

import sys
sys.path.append('../..')

def sigmoid(x): 
    z = 1/(1 + np.exp(-x)) 
    return z

''' ROC CURVE '''
def plot_roc(y_pred, y_true, roc_name, plot_dir, plot=True):
    # given the test_ground_truth, and test_predictions 
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    roc_auc = auc(fpr, tpr)
    roc_path=roc_name+".png"
    if plot:
        sns.set_style('white')
        sns.set_palette('Set1')

        # Create a figure with high resolution (300 dpi)
        fig, ax = plt.subplots(dpi=300)

        # Set the title with a fancy font
        ax.set_title(roc_name,  fontsize=16)

        # Plot the ROC curve with a smooth line and gradient fill
        ax.plot(fpr, tpr, color='#5C5D9E', linewidth=2, label='AUC = %.2f' % roc_auc)
        ax.fill_between(fpr, tpr, color='#5C5D9E', alpha=0.3)

        # Add a legend and set its position
        ax.legend(loc='lower right')

        # Add a dashed red line to represent the baseline
        ax.plot([0, 1], [0, 1], '--', color='#707071', linewidth=1)

        # Set the x-axis and y-axis limits
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        # Set the x-axis and y-axis labels with a fancy font
        ax.set_xlabel('False Positive Rate',  fontsize=12)
        ax.set_ylabel('True Positive Rate',  fontsize=12)

        # Customize tick labels with a fancy font
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)

        # Add a background grid for a fancy look
        ax.grid(color='lightgray', linestyle='--', linewidth=0.5)

        # Save the plot with a high-resolution output
        plt.savefig(f"{plot_dir}" + roc_path, bbox_inches='tight')
    return fpr, tpr, thresholds, roc_auc

# J = TP/(TP+FN) + TN/(TN+FP) - 1 = tpr - fpr
def choose_operating_point(fpr, tpr, thresholds):
    sens = 0
    spec = 0
    J = 0
    for _fpr, _tpr in zip(fpr, tpr):
        if _tpr - _fpr > J:
            sens = _tpr
            spec = 1-_fpr
            J = _tpr - _fpr
    return sens, spec

''' PRECISION-RECALL CURVE '''
def plot_pr(y_pred, y_true, pr_name, plot_dir, plot=True):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    # plot the precision-recall curves
    baseline = len(y_true[y_true==1]) / len(y_true)
    pr_path = pr_name+".jpg"
    if plot: 
        sns.set_style('whitegrid')
        sns.set_palette('Set2')

        # Set the font style
        #plt.rcParams['font.family'] = 'Arial'

        # Create a figure with high resolution (300 dpi)
        fig, ax = plt.subplots(dpi=300)

        # Set the title with a cool font
        ax.set_title(pr_name,  fontsize=16)

        # Plot the precision-recall curve with a customized line style
        ax.plot(recall, precision, color='#5C5D9E', linestyle='-', linewidth=2, label='AUC = %.2f' % pr_auc)

        # Add a legend and set its position
        ax.legend(loc='lower right')

        # Add a dashed red line to represent the baseline
        ax.plot([0, 1], [baseline, baseline], '--', color='#707071', linewidth=1)

        # Set the x-axis and y-axis limits
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        # Set the x-axis and y-axis labels with a cool font
        ax.set_xlabel('Recall',  fontsize=12)
        ax.set_ylabel('Precision',  fontsize=12)

        # Customize tick labels with a cool font
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)

        # Save the plot with a high-resolution output
        plt.savefig(f"{plot_dir}" + pr_path, bbox_inches='tight')
    return precision, recall, thresholds

def evaluate_internal(y_pred, y_true, cxr_labels, plot_dir,
                   roc_name='Receiver Operating Characteristic', pr_name='Precision-Recall Curve', label_idx_map=None):
    import warnings
    warnings.filterwarnings('ignore')

    num_classes = y_pred.shape[-1] # number of total labels

    dataframes = []
    # print(num_classes)
    counter=0
    for i in range(num_classes):

        if label_idx_map is None:
            y_pred_i = y_pred[:, i] # (num_samples,)
            y_true_i = y_true[:, i] # (num_samples,)

        else:
            y_pred_i = y_pred[:, i] # (num_samples,)

            true_index = label_idx_map[cxr_labels[i]]
            y_true_i = y_true[:, true_index] # (num_samples,)

        cxr_label = cxr_labels[i]
        counter = counter + 1

        ''' ROC CURVE '''
        roc_name = cxr_label + ' ROC Curve'
        # print(y_pred_i.shape)
        # print(y_true_i.shape)
        fpr, tpr, thresholds, roc_auc = plot_roc(y_pred_i, y_true_i, roc_name, plot_dir, plot=False)
        df = pd.DataFrame([roc_auc], columns=[cxr_label+'_auc'])
        dataframes.append(df)
        sens, spec = choose_operating_point(fpr, tpr, thresholds)

        ''' PRECISION-RECALL CURVE '''
        pr_name = cxr_label + ' Precision-Recall Curve'
        precision, recall, thresholds = plot_pr(y_pred_i, y_true_i, pr_name, plot_dir, plot=False)
        """
        results = [precision[0]]
        df = pd.DataFrame(results, columns=[cxr_label+'_precision'])
        dataframes.append(df)
        """
    dfs = pd.concat(dataframes, axis=1)
    return dfs

