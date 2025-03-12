import pandas as pd

def process_csv(input_csv_path: str, output_csv_path: str):
    # Load CSV into a Pandas DataFrame
    df = pd.read_csv(input_csv_path)
    
    # Ensure necessary columns exist
    if 'pid' not in df.columns or 'sct_ab_desc' not in df.columns:
        raise ValueError("Input CSV must contain 'pid' and 'sct_ab_desc' columns.")
    
    # Group by 'pid' and check if any value in 'sct_ab_desc' is 60
    has_cvd_mapping = df.groupby('pid')['sct_ab_desc'].apply(lambda x: 1 if 60 in x.values else 0)
    # TODO: once you have other report on the death reason, wrangle the labels in here

    result_df = has_cvd_mapping.reset_index(name='has_cvd')
    # Create a new DataFrame with only 'pid' and 'has_cvd'
    
    # Print number of unique pid
    print(f"Number of unique pid: {result_df['pid'].nunique()}")
    
    # Print number of 1s in has_cvd column
    print(f"Number of pids labeled as 1: {result_df['has_cvd'].sum()}")

    # Save the resulting dataframe to the output CSV file
    result_df.to_csv(output_csv_path, index=False)
    print(f"Processed file saved to: {output_csv_path}")


def merge_subject_labels_with_instance_vol(instance_vol_csv_path: str, subject_label_csv_path: str, output_csv_path: str):
    """
    Broadcast the subject labels to its instance volume based on 'pid'.
    """
    # Load instance volume and subject label data
    instance_df = pd.read_csv(instance_vol_csv_path)
    subject_label_df = pd.read_csv(subject_label_csv_path)
    
    # Ensure 'pid' column exists in both DataFrames
    if 'pid' not in instance_df.columns or 'pid' not in subject_label_df.columns:
        raise ValueError("Both CSV files must contain a 'pid' column.")
    
    # Merge the subject labels with instance volume data
    merged_df = instance_df.merge(subject_label_df, on='pid', how='left')
    before = merged_df['has_cvd'].isna().sum()
    
    # Fill missing has_cvd values with -1
    merged_df['has_cvd'] = merged_df['has_cvd'].fillna(-1)
    after = (merged_df['has_cvd'] == -1).sum()
    
    # sanity check
    assert before == after

    # Save the merged DataFrame to the output CSV file
    merged_df.to_csv(output_csv_path, index=False)
    
    print(f"Merged file saved to: {output_csv_path}")
if __name__ == '__main__':
    # process_csv(
    #     '/mnt/g/NLST/manifest-NLST_allCT/nlst_780_ctab_idc_20210527.csv',
    #     '/mnt/g/NLST/manifest-NLST_allCT/subject_cvd_significant_abnormal.csv'
    # )

    # merge the subject label to its instance CT vol
    merge_subject_labels_with_instance_vol(
        instance_vol_csv_path='/mnt/g/NLST/manifest-NLST_allCT/NLST_data_split_from_CVD_risk_estimator.csv', 
        subject_label_csv_path='/mnt/g/NLST/manifest-NLST_allCT/subject_cvd_significant_abnormal.csv',
        output_csv_path='/mnt/g/NLST/manifest-NLST_allCT/NLST_data_split_from_CVD_risk_estimator_with_cvd_abnormal_labels.csv'
    )