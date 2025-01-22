"""
main script to process the mimic data so that it matches the style of the ctrate data
    - a label file for each xray volume
    - a report csv file

"""
import os
import pandas as pd

#%%combine all the patholgies csv file into a single file
def merge_labels():
    # find all the relevant csv files
    labels_dir = "/Users/maxxyouu/Desktop/CT-CLIP/mimic-ct-raw/CT"
    csv_files = [os.path.join(labels_dir, file) for file in os.listdir(labels_dir) if file.endswith('.csv')]

    print(csv_files)
    print('total number of csv files:', len(csv_files))

    # generate a label file with column for each pathology and label
    combined_df = pd.DataFrame()

    # Function to extract the value for the disease type from the CTscan_labeldict
    def extract_value(row, disease):
        if pd.isna(row):  # Handle NaN
            return 0
        try:
            # Convert the string representation of the dictionary to an actual dictionary
            label_dict = eval(row) if isinstance(row, str) else row

            # handle halluciation cases
            if disease not in label_dict:
                return 0
            
            # label as 1 only by the following cases, otherwise we assume it is 0
            if label_dict[disease] == '1' or label_dict[disease] == 1 or str(label_dict[disease]).lower() == 'yes':
                return 1

            return 0

        except Exception as e:
            print(f"Error processing row: {row}, Error: {e}, assuming 0")
            return 0

    # Read and process each CSV file
    infected_counts = {}
    csv_file_row_counts = {}
    for csv_file in csv_files:
        # Extract the disease name from the filename
        disease_name = os.path.basename(csv_file).replace('CTscanlabels_type_singe_', '').replace('.csv', '')
        
        # Read the CSV file
        df = pd.read_csv(csv_file, usecols=['hadm_id', 'text', 'CTscan_labeldict'])

        unique_hadm_ids = df['hadm_id'].nunique()
        total = len(df)
        if unique_hadm_ids < total:
            print(f'non unique hadm_id in {disease_name}: {unique_hadm_ids} vs {total}')
            # assert False

        # Extract the label for each row of the DataFrame
        df[disease_name] = df['CTscan_labeldict'].apply(lambda x: extract_value(x, disease_name))
        
        # count number of cases that are 1 and number of rows for each csv file
        infected_counts[disease_name] = df[disease_name].value_counts().get(1, 0).item()
        csv_file_row_counts[disease_name] = len(df)

        # Drop the original CTscan_labeldict column
        df = df.drop(columns=['CTscan_labeldict'])
        
        # Merge with the combined DataFrame
        if combined_df.empty:
            combined_df = df
        else:
            # TODO: double check this.
            combined_df = pd.merge(combined_df, df, on=['hadm_id', 'text'], how='outer') # this should take care of multiple texts for the same hadm_id

    # print label counts
    print('==================== summary ====================')
    print('disease case for each label:', infected_counts)
    print('total number of cases with at least one disease: ', sum([infected_counts[key] for key in infected_counts]))
    print('row counts for each disease: ', csv_file_row_counts)
    print('row counts: ', sum([csv_file_row_counts[key] for key in csv_file_row_counts]))
    print('number of null after merge: ', combined_df.isnull().sum().sum())
    print('number of nan after merge: ', combined_df.isna().sum().sum())
    print('number of unique hadm_id: ', combined_df['hadm_id'].nunique())

    unique_hadm_id_text = df[['hadm_id', 'text']].drop_duplicates().shape[0]
    print('number of unique [hadm_id,text]: ', unique_hadm_id_text)


    # if the following is true => label is consistent with different report
    filtered_combined_df = combined_df.drop_duplicates().sort_values(by='hadm_id')
    assert unique_hadm_id_text == len(filtered_combined_df) # check if there are any duplicates left based on the two keys

    # based on the combined dataframe, seperate the report into different section like ctrate report
    CLINICAL_INFORMATION = 'ClinicalInformation_EN'
    TECHNIQUE = 'Technique_EN'
    FINIDNGS = 'Findings_EN'
    IMPRESSIONS = 'Impressions_EN'
    HADM_ID = 'hadm_id'
    NOT_GIVEN = 'Not given.'
    temp_df = pd.DataFrame(columns=[CLINICAL_INFORMATION, TECHNIQUE, FINIDNGS, IMPRESSIONS])

    for index, row in filtered_combined_df.iterrows():
        text = row['text']
        hadm_id = row[HADM_ID]

        impression = NOT_GIVEN
        if 'impression:' in text.lower():
            impression = text[text.lower().index('impression:') + len('impression:'):]
            text = text[:text.lower().index('impression:')] # remove the impression part so it is easier for ealier section
        elif 'impressions:' in text.lower():
            impression = text[text.lower().index('impressions:') + len('impressions:'):]
            text = text[:text.lower().index('impressions:')]

        findings = NOT_GIVEN
        if 'findings:' in text.lower():
            findings = text[text.lower().index('findings:') + len('findings:'):]
            text = text[:text.lower().index('findings:')] # remove the finding part so it is easier for ealier section
        elif 'finding:' in text.lower():
            findings = text[text.lower().index('finding:') + len('finding:'):]
            text = text[:text.lower().index('finding:')]
        
        technique = NOT_GIVEN
        if 'technique:' in text.lower():
            technique = text[text.lower().index('technique:') + len('technique:'):]
            text = text[:text.lower().index('technique:')] # remove the technique part so it is easier for ealier section
        elif 'techniques:' in text.lower():
            technique = text[text.lower().index('techniques:') + len('techniques:'):]
            text = text[:text.lower().index('techniques:')]

        history = ''
        if 'history:' in text.lower():
            history = text[text.lower().index('history:') + len('history:'):]
            text = text[:text.lower().index('history:')] # remove the history part so it is easier for ealier section
        elif 'histories:' in text.lower():
            history = text[text.lower().index('histories:') + len('histories:'):]
            text = text[:text.lower().index('histories:')]
        elif 'historys:' in text.lower():
            history = text[text.lower().index('historys:') + len('historys:'):]
            text = text[:text.lower().index('historys:')]

        indication = ''
        if 'indications:' in text.lower():
            indication = text[text.lower().index('indications:') + len('indications:'):]
            text = text[:text.lower().index('indications:')]
        elif 'indication:' in text.lower():
            indication = text[text.lower().index('indication:') + len('indication:'):]
            text = text[:text.lower().index('indication:')]

        # combine both indication and history into clinical information
        clinical_informaiton = NOT_GIVEN
        if (indication + history) != '':
            clinical_informaiton = indication + history

        row_info = {CLINICAL_INFORMATION: clinical_informaiton.strip(), TECHNIQUE: technique.strip(), FINIDNGS: findings.strip(), IMPRESSIONS: impression.strip()}

        new_row = pd.DataFrame([row_info])
        temp_df = pd.concat([temp_df, new_row], ignore_index=True)

    # join the colums the two dataframes
    temp_df = temp_df.reset_index(drop=True)
    filtered_combined_df = filtered_combined_df.reset_index(drop=True)
    merged_df = pd.concat([filtered_combined_df, temp_df], axis=1)
    print('number of rows after concatenate the two dfs: ', len(merged_df))

    # Save the combined DataFrame to a single CSV file
    output_file = "/Users/maxxyouu/Desktop/CT-CLIP/mimic-ct-raw/mimic_ct_predicted_labels.csv"
    merged_df.to_csv(output_file, index=False)

    print(f"Combined CSV file saved as {output_file}")
    print('DONE')

#%% find the INTERSECTION the two csv files based on the hadm_id and discharge_hadm_id

# Read the CSV files into DataFrames
predicted_labels_file_path = "/Users/maxxyouu/Desktop/CT-CLIP/mimic-ct-raw/mimic_ct_predicted_labels.csv"
base_file_path = '/Users/maxxyouu/Desktop/CT-CLIP/mimic-ct-raw/final_mimiccxr-meta_with_view_total_df_with_cxr_path.csv'
labels_df = pd.read_csv(predicted_labels_file_path)
base_df = pd.read_csv(base_file_path)

# Drop rows with empty 'discharge_hadm_id' in csv2
base_df = base_df[base_df['discharge_hadm_id'].notna()]

# Ensure 'hadm_id' in csv1 has no empty values (SANITY CHECK)
if labels_df['hadm_id'].isna().any():
    raise ValueError("The 'hadm_id' column in csv1 contains empty values.")

# Print the number of rows in csv1 with non-empty 'discharge_hadm_id'
non_empty_rows_base_df = len(base_df[base_df['discharge_hadm_id'].notna()])
print(f"Number of rows in base csv file with non-empty 'discharge_hadm_id': {non_empty_rows_base_df}")

# Print the number of unique discharge_hadm_id
unique_discharge_hadm_id = base_df['discharge_hadm_id'].nunique()
print(f'Number of unique discharge_hadm_id: {unique_discharge_hadm_id}')

#NOTE: unique_discharge_hadm_id is smaller than non_empty_rows_base_df => multiple cxr for one hadm_id 1925 vs 3562

# Perform the inner join to pair up the predicted labels and the ct report
result = pd.merge(labels_df, base_df, left_on='hadm_id', right_on='discharge_hadm_id', how='inner').sort_values(by=['hadm_id', 'discharge_hadm_id'])

print('size for unique (hadm_id and discharge_hadm_id): ', result[['hadm_id', 'discharge_hadm_id']].drop_duplicates().shape[0])

# Save the result to a new CSV file
paired_mimic_ct_report_file = "/Users/maxxyouu/Desktop/CT-CLIP/mimic-ct-raw/mimic_ct_report_paired.csv"

result.to_csv(paired_mimic_ct_report_file, index=False)

# Print the number of rows in the resulting CSV file
print(f"Inner join complete. Result saved to 'inner_join_result.csv'. Number of rows: {len(result)}")

#%% drop the labels for the training and validation split on the ct rate dataset and save as a new file..
dropping_pathos = ['Medical material',
                'Cardiomegaly', 
                'Lung nodule',
                'Lung opacity', 
                'Pulmonary fibrotic sequela', 
                'Pleural effusion', 
                'Consolidation']

# Path to the CSV file
# input_csv_path = "/Users/maxxyouu/Desktop/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_train_predicted_labels.csv"  # Replace with your actual file path
# output_csv_path = "/Users/maxxyouu/Desktop/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_train_mimic_labels.csv"  # Replace with the desired output file path
input_csv_path = "/Users/maxxyouu/Desktop/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_valid_predicted_labels.csv"  # Replace with your actual file path
output_csv_path = "/Users/maxxyouu/Desktop/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_valid_mimic_labels.csv"  # Replace with the desired output file path

# Load the CSV file into a DataFrame
df = pd.read_csv(input_csv_path)

# Remove the specified columns
updated_df = df.drop(columns=dropping_pathos, errors='ignore')

# Save the updated DataFrame to a new CSV file
updated_df.to_csv(output_csv_path, index=False)

print(f"Updated CSV saved to {output_csv_path}.")

#%% rearrange the csv column so that it matches the following order
# should be the intersection of the following labels, the one labeled with # means the intersection
pathologies = ['Arterial wall calcification', #
                'Pericardial effusion', #
                'Coronary artery wall calcification', #
                'Hiatal hernia', #
                'Lymphadenopathy', #
                'Emphysema', #
                'Atelectasis', #
                'Mosaic attenuation pattern',#
                'Peribronchial thickening', #
                'Bronchiectasis', #
                'Interlobular septal thickening']#

input_csv_path = '/Users/maxxyouu/Desktop/CT-CLIP/mimic-ct-raw/mimic_ct_report_paired.csv'

output_csv_path = '/Users/maxxyouu/Desktop/CT-CLIP/dataset/multi_abnormality_labels/mimic_ct_report_paired_with_ordered_label.csv' # NOTE: this is the ultimate file.
label_only_csv_path = '/Users/maxxyouu/Desktop/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_external_valid_mimic_labels.csv'
df = pd.read_csv(input_csv_path)

# Arrange columns as per the given order, and append the rest of the columns
ordered_columns = [col for col in pathologies if col in df.columns]  # Retain only columns that exist in the DataFrame
assert len(ordered_columns) == len(pathologies)

remaining_columns = [col for col in df.columns if col not in ordered_columns]
final_columns = ordered_columns + remaining_columns

def resolve_duplicate_hadm_ids(df, output_path):
    # Ensure the column exists
    if 'hadm_id' not in df.columns:
        raise ValueError("The input CSV file does not contain a 'hadm_id' column.")

    # Create a dictionary to count occurrences of each hadm_id
    hadm_id_counts = {}

    # List to store the new hadm_id values
    new_hadm_ids = []

    # Iterate over the hadm_id column
    for hadm_id in df['hadm_id']:
        if hadm_id not in hadm_id_counts:
            hadm_id_counts[hadm_id] = 0
        
        hadm_id_counts[hadm_id] += 1
        
        # Append the suffix for all hadm_id values
        new_hadm_ids.append(f"{hadm_id}.{hadm_id_counts[hadm_id]}")

    # Replace the hadm_id column with the updated values
    df['hadm_id'] = new_hadm_ids

    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_path, index=False)

    print(f"Processed file saved to {output_path}")

# apply the above function for the following files.

# Reorder the DataFrame
df = df[final_columns]
# Save the updated DataFrame to a new CSV file
resolve_duplicate_hadm_ids(df, output_csv_path) # NOTE: should apply only once.
# df.to_csv(output_csv_path, index=False)
print(f"Columns rearranged and updated CSV saved to {output_csv_path}.")

#%% get the label files
# NOTE: double check this, the duplicate should be resolved.
# output_csv_path = '/Users/maxxyouu/Desktop/CT-CLIP/dataset/multi_abnormality_labels/mimic_ct_report_paired_with_ordered_label.csv'
# df = pd.read_csv(output_csv_path)
# selected_columns = ['hadm_id'] + [col for col in pathologies if col in df.columns]
# pathologies_df = df[selected_columns]
# pathologies_df.to_csv(label_only_csv_path, index=False)
# print(f"Pathologies and 'hadm_id' columns saved to {label_only_csv_path} as a label csv file.")


#%% generate a report csv file following the same format as the ctrate report csv file
# CLINICAL_INFORMATION = 'ClinicalInformation_EN'
# TECHNIQUE = 'Technique_EN'
# FINIDNGS = 'Findings_EN'
# IMPRESSIONS = 'Impressions_EN'
# HADM_ID = 'hadm_id'

# report_cols = [HADM_ID, CLINICAL_INFORMATION, TECHNIQUE, FINIDNGS, IMPRESSIONS]
# input_csv_path = '/Users/maxxyouu/Desktop/CT-CLIP/dataset/multi_abnormality_labels/mimic_ct_report_paired_with_ordered_label.csv'
# output_csv_path = '/Users/maxxyouu/Desktop/CT-CLIP/dataset/radiology_text_reports/external_valid_mimic_report.csv'
# df = pd.read_csv(input_csv_path)

# selected_report = df[report_cols]
# selected_report.to_csv(output_csv_path, index=False)
# print(f"report CSV saved to {output_csv_path}.")

