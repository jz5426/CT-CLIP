"""
mainly for the ablation study zero-shot where when text_cl = 1 it has one excel file and ct_cl = 1 it has another excel file, this mainly to combine them for ease of analysis.
"""

import pandas as pd

# Load the two Excel files
df1 = pd.read_excel('/cluster/home/t135419uhn/CT-CLIP/shell_scripts/experiment_results/ablation_zero_shot.xlsx')
df2 = pd.read_excel('/cluster/home/t135419uhn/CT-CLIP/shell_scripts/experiment_results/ablation_zero_shot_another.xlsx')

# Concatenate the dataframes row-wise
combined_df = pd.concat([df1, df2], ignore_index=True)

# Save the combined dataframe to a new Excel file
output_path = '/cluster/home/t135419uhn/CT-CLIP/shell_scripts/experiment_results/ablation_zero_shot.xlsx'
combined_df.to_excel(output_path, index=False)

print(f"Combined file saved as {output_path}")