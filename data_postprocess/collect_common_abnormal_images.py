import pandas as pd
import os
import random
import shutil

def get_ids_with_disease(csv_file, parent_dir, disease_column, id_column, extension, dataset):
    """
    Returns a list of IDs from the given CSV file where the specified disease column has a value of 1.

    Parameters:
    - csv_file (str): Path to the CSV file.
    - disease_column (str): Name of the disease column to filter on.
    - id_column (str): Name of the column containing IDs.

    Returns:
    - List of IDs where the disease column has value 1.
    """
    df = pd.read_csv(csv_file)

    # Ensure columns exist
    if disease_column not in df.columns:
        raise ValueError(f"Disease column '{disease_column}' not found in CSV.")
    if id_column not in df.columns:
        raise ValueError(f"ID column '{id_column}' not found in CSV.")

    # Filter and get IDs
    filtered_ids = df[df[disease_column] == 1][id_column].tolist()

    if dataset == 'mimic':
        return [os.path.join(parent_dir, _id+extension) for _id in filtered_ids if os.path.exists(os.path.join(parent_dir, _id+extension))]
    

    def map_filename_to_path(filename):
        """
        Convert a filename like 'train_3_a_1.nii.gz' to a path like 'train_3/train_3a/train_3_a_1.mha'.
        
        Parameters:
        - filename (str): The original filename.
        
        Returns:
        - str: The mapped path.
        """
        # Remove extension
        if filename.endswith(".nii.gz"):
            base = filename[:-7]
        else:
            raise ValueError("Filename must end with .nii.gz")

        parts = base.split('_')
        if len(parts) < 3:
            raise ValueError("Filename must have at least 3 underscore-separated parts.")

        dir1 = '_'.join(parts[:2])  # e.g., train_3
        dir2 = parts[0] + '_' + parts[1] + parts[2]  # e.g., train_3a
        new_filename = base + extension

        return os.path.join(dir1, dir2, new_filename)

    if dataset == 'ct-rate':
        results = []
        filtered_ids = random.choices(filtered_ids, k=100)
        for _id in filtered_ids:
            subdir = map_filename_to_path(_id)
            if os.path.exists(os.path.join(parent_dir, subdir)):
                results.append(os.path.join(parent_dir, subdir))
        return results
    

def copy_files_to_directory(file_paths, destination_dir):
    """
    Copies a list of files to the specified destination directory.
    Creates the destination directory if it doesn't exist.

    Parameters:
    - file_paths (list of str): List of full paths to the source files.
    - destination_dir (str): Path to the destination directory.
    """
    # Create destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    for file_path in file_paths:
        if os.path.isfile(file_path):
            shutil.copy(file_path, destination_dir)
        else:
            print(f"Warning: File not found and skipped - {file_path}")

if __name__ == '__main__':
    # corresponding dataset locates in /cluster/projects/mcintoshgroup/publicData/CT-RATE/preprocessed_mimic/mimic_preprocessed_xray_mha
    mimic_cawc_ids = get_ids_with_disease(
        "/cluster/projects/mcintoshgroup/publicData/CT-RATE/dataset/multi_abnormality_labels/mimic_ct_report_paired_with_ordered_label_pa_ap.csv", # to get the labels
        "/cluster/projects/mcintoshgroup/publicData/CT-RATE/preprocessed_mimic/mimic_preprocessed_xray_rgb", # where the files are lcoated under
        "Coronary artery wall calcification", 
        "hadm_id",
        '.png',
        'mimic'
        )
    print(f"Found {len(mimic_cawc_ids)} patients with Coronary artery wall calcification.")

    ctrate_cawc_ids = get_ids_with_disease(
        "/cluster/projects/mcintoshgroup/publicData/CT-RATE/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_train_predicted_labels.csv", # to get the labels
        "/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/train_preprocessed_xray_mha", # where the files are lcoated under
        "Coronary artery wall calcification", 
        "VolumeName",
        '.mha',
        'ct-rate'
        )
    print(f"Found {len(ctrate_cawc_ids)} patients with Coronary artery wall calcification.")

    copy_files_to_directory(mimic_cawc_ids, '/cluster/projects/mcintoshgroup/publicData/CT-RATE/coronary_artery_wall_cal_visualize/mimic')
    copy_files_to_directory(ctrate_cawc_ids, '/cluster/projects/mcintoshgroup/publicData/CT-RATE/coronary_artery_wall_cal_visualize/CT-RATE')
    print('done')


