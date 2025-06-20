import pandas as pd

def create_risk_labels(raw_scores_file, output_scores_file):
    # Load your Excel file
    df = pd.read_excel(raw_scores_file)  # replace with your actual file path

    # Define risk mapping function
    def value_to_risk(value):
        if value <= 10:
            return 0
        elif value <= 100:
            return 1
        elif value <= 400:
            return 2
        else:
            return 3

    # List of target columns to convert
    target_columns = ["LCA", "LAD", "LCX", "RCA", "total"]

    # Apply risk mapping to each column and create new risk columns
    for col in target_columns:
        df[f"{col}_risk"] = df[col].apply(value_to_risk)

    # Save to a new Excel file (optional)
    df.to_excel(output_scores_file, index=False)

    # Display result (optional)
    print(df.head())

def find_risk_factor_distribution(label_scores_file):
    # Load the file (if not already loaded)
    df = pd.read_excel(label_scores_file)  # or your original file if risk columns already exist

    # Risk columns to analyze
    risk_columns = ["LCA_risk", "LAD_risk", "LCX_risk", "RCA_risk", "total_risk"]

    # Compute and display the distribution of each risk level (0–3) per column
    for col in risk_columns:
        print(f"Distribution for {col}:")
        print(df[col].value_counts().sort_index())  # sort_index ensures output in order 0,1,2,3

if __name__ == '__main__':
    raw_scores_file = "/cluster/projects/mcintoshgroup/publicData/coca/cocacoronarycalciumandchestcts-2/deidentified_nongated/scores.xlsx"
    label_scores_file = "/cluster/projects/mcintoshgroup/publicData/coca/cocacoronarycalciumandchestcts-2/deidentified_nongated/scores_labels.xlsx"
    # create_risk_labels(raw_scores_file, label_scores_file)

    find_risk_factor_distribution(label_scores_file)

