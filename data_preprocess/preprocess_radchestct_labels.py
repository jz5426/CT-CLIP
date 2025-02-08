import pandas as pd
import sys

def load_excel(file_path):
    """Load an Excel file into a pandas DataFrame."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        sys.exit(1)

def check_columns_consistency(dfs):
    """Check if all DataFrames have the same set of columns."""
    column_sets = [set(df.columns) for df in dfs]
    if all(columns == column_sets[0] for columns in column_sets):
        return True
    else:
        print("Error: The input files do not have the same set of columns.")
        sys.exit(1)

def merge_excels(file_paths):
    """Merge multiple Excel files if they have the same columns."""
    dfs = [load_excel(file) for file in file_paths]

    # Check if all files have the same columns
    if check_columns_consistency(dfs):
        merged_df = pd.concat(dfs, ignore_index=True)
        return merged_df
    return None

def process_labels(disease_names, df):
    # Initialize an empty DataFrame to store the results
    result_df = pd.DataFrame()
    result_df['NoteAcc_DEID'] = df['NoteAcc_DEID']

    disease_distribution = {}
    # Iterate over each disease name
    for disease in disease_names:
        # Find columns that contain the disease keyword (case-insensitive)
        disease_columns = [col for col in df.columns if disease.lower() in col.lower()]
        
        # Filter columns that contain only 0, 0.0, 1, or 1.0
        valid_columns = []
        for col in disease_columns:
            if all(df[col].isin([0, 0.0, 1, 1.0])):
                valid_columns.append(col)
        assert len(valid_columns) == len(disease_columns)

        # If valid columns are found, merge them by taking the max value
        if valid_columns:
            result_df[disease.lower()] = df[valid_columns].max(axis=1)
            disease_distribution[disease] = sum(df[valid_columns].max(axis=1))

    print(disease_distribution)
    return result_df

if __name__ == "__main__":
    # Example file paths (update these with actual file paths)
    file_paths = [
        "/cluster/projects/mcintoshgroup/publicData/RADChestCT/imgtest_Abnormality_and_Location_Labels.csv",
        "/cluster/projects/mcintoshgroup/publicData/RADChestCT/imgtrain_Abnormality_and_Location_Labels.csv",
        "/cluster/projects/mcintoshgroup/publicData/RADChestCT/imgvalid_Abnormality_and_Location_Labels.csv"
    ]
    merge_labels_output_path = "/cluster/projects/mcintoshgroup/publicData/RADChestCT/merged_original_labels.csv"
    final_labels_output_path = "/cluster/projects/mcintoshgroup/publicData/RADChestCT/final_labels_pure.csv"
    merged_labels_df = merge_excels(file_paths)
    if merged_labels_df is not None:
        # merged_labels_df.to_csv(merge_labels_output_path, index=False)
        # print(f"Merged file saved as: {merge_labels_output_path}")
        # path_col_names = [
        #     'calcification',
        #     'Cardiomegaly',
        #     'pericardial_effusion',
        #     'hernia',
        #     'Lymphadenopathy',
        #     'Emphysema',
        #     'Atelectasis',
        #     'nodule',
        #     'opacity',
        #     'fibrosis',
        #     'pleural_effusion',
        #     'bronchial_wall_thickening', # assumed
        #     'Consolidation',
        #     'Bronchiectasis',
        #     'septal_thickening'
        # ]
        path_col_names = [
            'calcification',
            'pericardial_effusion',
            'hernia',
            'lymphadenopathy',
            'emphysema',
            'fibrosis',
            'bronchial_wall_thickening',
            'bronchiectasis',
            'septal_thickening'
        ]
        merged_label_frames = process_labels(path_col_names, merged_labels_df)

        # Save the new DataFrame to a CSV file
        merged_label_frames.to_csv(final_labels_output_path, index=False)
        print(f'final label file saved as: {final_labels_output_path}')
    else:
        print('fail to merge the label files => inconsistent columns')
        

