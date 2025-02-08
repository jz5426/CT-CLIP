import pandas as pd

def merge_disease_labels(csv_file, radchest_ctrate_label_mappings):
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Ensure 'VolumeName' is retained as the first column
    if 'VolumeName' in df.columns:
        volume_name_col = df[['VolumeName']]
    else:
        raise ValueError("Column 'VolumeName' not found in the input CSV file.")
    
    # Identify columns to keep based on mapping values
    disease_columns = set(sum(radchest_ctrate_label_mappings.values(), []))
    df = df[[col for col in df.columns if col in disease_columns]]
    
    # Create a new dataframe to store the merged results
    merged_df = pd.DataFrame()
    
    # Merge disease labels based on the mapping
    for new_col, old_cols in radchest_ctrate_label_mappings.items():
        existing_cols = [col for col in old_cols if col in df.columns]
        merged_df[new_col] = df[existing_cols].max(axis=1) if existing_cols else 0
    
    # Concatenate 'VolumeName' with merged results
    merged_df = pd.concat([volume_name_col, merged_df], axis=1)
    
    return merged_df


if __name__ == '__main__':
    # radchest_ctrate_label_mappings = {
    #     'calcification': ['Arterial wall calcification', 'Coronary artery wall calcification'],
    #     'cardiomegaly': ['Cardiomegaly'],
    #     'pericardial_effusion':['Pericardial effusion'],
    #     'hernia': ['Hiatal hernia'],
    #     'lymphadenopathy': ['Lymphadenopathy'],
    #     'emphysema': ['Emphysema'],
    #     'atelectasis': ['Atelectasis'],
    #     'nodule': ['Lung nodule'],
    #     'opacity': ['Lung opacity'],
    #     'fibrosis': ['Pulmonary fibrotic sequela'],
    #     'pleural_effusion': ['Pleural effusion'],
    #     'bronchial_wall_thickening': ['Peribronchial thickening'], # assumed
    #     'consolidation': ['Consolidation'],
    #     'bronchiectasis': ['Bronchiectasis'],
    #     'septal_thickening': ['Interlobular septal thickening'],
    # }

    # radchest_ctrate_labels = merge_disease_labels(
    #     '/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_train_predicted_labels.csv',
    #     radchest_ctrate_label_mappings
    # )
    # radchest_ctrate_labels.to_csv(
    #     '/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_train_radchest_ct_labels.csv',
    #     index=False
    # )

    radchest_ctrate_label_mappings = {
        'calcification': ['Arterial wall calcification', 'Coronary artery wall calcification'],
        'pericardial_effusion':['Pericardial effusion'],
        'hernia': ['Hiatal hernia'],
        'lymphadenopathy': ['Lymphadenopathy'],
        'emphysema': ['Emphysema'],
        'fibrosis': ['Pulmonary fibrotic sequela'],
        'bronchial_wall_thickening': ['Peribronchial thickening'], # assumed
        'bronchiectasis': ['Bronchiectasis'],
        'septal_thickening': ['Interlobular septal thickening'],
    }

    radchest_ctrate_labels = merge_disease_labels(
        '/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_train_predicted_labels.csv',
        radchest_ctrate_label_mappings
    )
    radchest_ctrate_labels.to_csv(
        '/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_train_radchest_ct_pure_labels.csv',
        index=False
    )
