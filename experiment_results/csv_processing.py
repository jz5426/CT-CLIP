import os
import pandas as pd
from glob import glob

def merge_csv(csv_folder, output_name):

    # Get all CSV files in the folder
    csv_files = glob(os.path.join(csv_folder, "*.csv"))


    # Read all CSV files into DataFrames and count rows
    dfs = []
    manual_row_count = 0

    for file in csv_files:
        df = pd.read_csv(file)
        row_count = df.shape[0]  # Number of rows in this file
        manual_row_count += row_count
        dfs.append(df)

    # Merge all DataFrames
    merged_df = pd.concat(dfs, ignore_index=True)

    # Get the total number of rows in the merged DataFrame
    merged_row_count = merged_df.shape[0]

    # Print row count verification
    print(f"Total rows in merged DataFrame: {merged_row_count}")
    print(f"Total rows manually counted from individual files: {manual_row_count}")

    # Double-check if they match
    if merged_row_count == manual_row_count:
        print("✅ Row counts match! The merge was successful.")
    else:
        print("❌ Mismatch detected! Check for errors in file reading or missing rows.")

    save_path = os.path.join(csv_folder, f"{output_name}.csv")
    merged_df.to_csv(save_path, index=False)

    return

def split_csv_by_group(input_csv, output_folder, group_by_attrs, output_columns=None, filter_func=None):
    """
    Reads a CSV file and splits it into multiple CSV files grouped by user-specified attributes, 
    with a user-defined selection of output columns and an optional row-wise filtering function.

    Parameters:
        input_csv (str): Path to the input CSV file.
        output_folder (str): Folder where the split CSV files will be saved.
        group_by_attrs (list of str): List of column names to group by.
        output_columns (list of str, optional): List of columns to keep in the output files.
                                                If None, all columns are kept.
        filter_func (function, optional): A function that takes a row (Series) and returns True if the row should be kept.
                                          If None, all rows are kept.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read the CSV file
    df = pd.read_csv(input_csv)

    # Ensure the provided group_by attributes exist in the DataFrame
    missing_group_attrs = [attr for attr in group_by_attrs if attr not in df.columns]
    if missing_group_attrs:
        raise ValueError(f"Columns not found in CSV file: {missing_group_attrs}")

    # If output_columns is specified, ensure they exist in the DataFrame
    if output_columns:
        missing_output_attrs = [col for col in output_columns if col not in df.columns]
        if missing_output_attrs:
            raise ValueError(f"Output columns not found in CSV file: {missing_output_attrs}")

    # Group by specified attributes
    grouped = df.groupby(group_by_attrs)

    # Iterate through each group and save as a new CSV file
    for group_keys, group in grouped:
        # Apply filtering function row-wise if provided
        if filter_func:
            group = group[group.apply(filter_func, axis=1)]
            if group.empty:
                continue  # Skip saving if the filtered group is empty

        # Select only specified output columns if provided
        if output_columns:
            group = group[output_columns]

        # Convert group keys into a string for filename
        filename = "_".join(map(str, group_keys)) + ".csv"
        filepath = os.path.join(output_folder, filename)

        # Save group to CSV
        group.to_csv(filepath, index=False)
        print(f"Saved: {filepath}")

# Example filter function: Only keep rows where AUC > 0.8 (Applied per row)
def filter_custom_models(row):
    if 'ins' in row['model'] or 'exp' in row['model']: # indicate custom models
        return True if 'infoNCE' in row['model'] else False
    return True

def filter_model_of_interest(row):
    """
    keep only the models that we most likely show in the paper
    """
    interested = [
        'swin_pretrained_exp_infoNCE',
        'resnet_pretrained_exp_infoNCE',
        'swin_exp_infoNCE',
        'resnet_exp_infoNCE'
    ]

    if 'ins' in row['model'] or 'exp' in row['model']: # indicate custom models
        return True if row['model'] in interested else False
    return True

if __name__ == '__main__':
    # merge_csv('/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/chris/CT-CLIP/experiment_results/linear_probing', 'linear_probe_global_results')

    # NOTE: only keep the models of interest
    input_csv = "/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/chris/CT-CLIP/experiment_results/linear_probing/linear_probe_global_results.csv"  # Replace with the actual file path
    output_folder = "/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/chris/CT-CLIP/experiment_results/linear_probing/group_by_dataset_modelsOfInterest/"  # Replace with desired output folder
    group_by_attrs = ["dataset", "few_shot"]  # Specify the attributes to group by
    output_columns = ["dataset", "model", "few_shot", "AUC", "PR_AUC"]  # Columns to include in output files
    filter_func = filter_model_of_interest
    split_csv_by_group(input_csv, output_folder, group_by_attrs, output_columns, filter_func)

    # NOTE: only keep the models of interest with details
    input_csv = "/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/chris/CT-CLIP/experiment_results/linear_probing/linear_probe_global_results.csv"  # Replace with the actual file path
    output_folder = "/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/chris/CT-CLIP/experiment_results/linear_probing/group_by_dataset_modelsOfInterest_Details/"  # Replace with desired output folder
    group_by_attrs = ["dataset", "few_shot"]  # Specify the attributes to group by
    output_columns = ["dataset", "model", "few_shot", "AUC", "PR_AUC", "labels", "pred_probs", "auc_per_class"]  # Columns to include in output files
    filter_func = filter_model_of_interest
    split_csv_by_group(input_csv, output_folder, group_by_attrs, output_columns, filter_func)
