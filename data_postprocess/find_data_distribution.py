"""
plot label distribution of the dataset
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_abnormality_distribution(csv_file, abnormality_col_range, img_name, img_title, remove_singletons=False):
    """
    Plots a histogram of abnormality distribution from the specified CSV file.
    
    Parameters:
    - csv_file (str): Path to the CSV file.
    - abnormality_col_range (tuple): Tuple (start_idx, end_idx) for abnormality columns.
    """
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Extract the columns for abnormalities
    start_idx, end_idx = abnormality_col_range
    abnormality_df = df.iloc[:, start_idx:end_idx]
    
    # Count the number of 1s for each abnormality
    abnormality_counts = (abnormality_df == 1).sum()
    
    # Optionally remove abnormalities with only 1 or 0 positives
    if remove_singletons:
        abnormality_counts = abnormality_counts[abnormality_counts > 1]

    # Sort by column name
    abnormality_counts = abnormality_counts.sort_index()

    # Set seaborn pastel palette
    sns.set(style="whitegrid")
    colors = sns.color_palette("pastel")
    
    # Plot the histogram
    plt.figure(figsize=(12, 6))
    abnormality_counts.plot(kind='bar')
    plt.xlabel("Abnormalities")
    plt.ylabel("Number of Positive Cases")
    plt.title(f"Distribution of Abnormalities for {img_title}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f'{img_name}.png')

if __name__ == '__main__':
    plot_abnormality_distribution(
        "/cluster/projects/mcintoshgroup/publicData/CT-RATE/dataset/multi_abnormality_labels/mimic_ct_report_paired_with_ordered_label_pa_ap.csv", 
        (0, 11),
        '/cluster/home/t135419uhn/CT-CLIP/data_postprocess/mimic_ct_label_distribution',
        'MIMIC-CT',
        remove_singletons=True
        )

    plot_abnormality_distribution(
        "/cluster/projects/mcintoshgroup/publicData/CT-RATE/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_train_predicted_labels.csv", 
        (1, 18),
        '/cluster/home/t135419uhn/CT-CLIP/data_postprocess/ct_rate_label_distribution',
        'CT-RATE',
        remove_singletons=False
        )