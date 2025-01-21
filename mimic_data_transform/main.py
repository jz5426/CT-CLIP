"""
main script to process the mimic data so that it matches the style of the ctrate data
    - a label file for each xray volume
    - a report csv file

"""
import os
import pandas as pd
import numpy as np
import re

## combine all the patholgies csv file into a single file

## find the intersection the two csv files based on the hadm_id and discharge_hadm_id

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

    # Extract the value for the specific disease
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
        combined_df = pd.merge(combined_df, df, on=['hadm_id', 'text'], how='outer')

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
# TODO:
CLINICAL_INFORMATION = 'ClinicalInformation_EN'
TECHNIQUE = 'Technique_EN'
FINIDNGS = 'Findings_EN'
IMPRESSIONS = 'Impressions_EN'
HADM_ID = 'hadm_id'
NOT_GIVEN = 'Not given.'
# temp_df = pd.DataFrame(columns=[HADM_ID, CLINICAL_INFORMATION, TECHNIQUE, FINIDNGS, IMPRESSIONS])
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

    # row_info = {HADM_ID: hadm_id, CLINICAL_INFORMATION: clinical_informaiton.strip(), TECHNIQUE: technique.strip(), FINIDNGS: findings.strip(), IMPRESSIONS: impression.strip()}
    row_info = {CLINICAL_INFORMATION: clinical_informaiton.strip(), TECHNIQUE: technique.strip(), FINIDNGS: findings.strip(), IMPRESSIONS: impression.strip()}

    new_row = pd.DataFrame([row_info])
    temp_df = pd.concat([temp_df, new_row], ignore_index=True)

# join the colums the two dataframes
temp_df = temp_df.reset_index(drop=True)
filtered_combined_df = filtered_combined_df.reset_index(drop=True)
merged_df = pd.concat([filtered_combined_df, temp_df], axis=1)
print('number of rows after inner join: ', len(merged_df))

# Save the combined DataFrame to a single CSV file
output_file = "/Users/maxxyouu/Desktop/CT-CLIP/mimic-ct-raw/mimic_ct_predicted_labels.csv"
merged_df.to_csv(output_file, index=False)

print(f"Combined CSV file saved as {output_file}")
print('DONE')


## generate a report csv file following the same format as the ctrate report csv file