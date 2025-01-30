import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score

def rename_pickle_files(directory):
    # Define the keywords to keep
    models = ["resnet", "swin", "densenet"]
    datasets = {"cxr_xray": "cxrClip", "medclip": "medclip", "gloria": "gloria"}
    pretrained_status = {"pretrained_true": "pretrainTrue", "pretrained_false": "pretrainFalse"}
    
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if not filename.endswith(".pkl"):  # Process only pickle files
            continue
        
        lower_filename = filename.lower()
        
        # Extract train portion and format it
        train_portion_match = re.search(r"train_portion_(\d*\.?\d+)_data", lower_filename)
        if not train_portion_match:
            assert False
            continue  # Skip files that don't have this required pattern
        train_portion = f"trainPortion{train_portion_match.group(1)}"
        
        # Find which model name exists in filename
        model_name = next((m for m in models if m in lower_filename), "")
        
        # Find which dataset name exists in filename and map it
        dataset_name = next((datasets[d] for d in datasets if d in lower_filename), "")
        
        # Find pretrained status if present and map it
        pretrained = next((pretrained_status[p] for p in pretrained_status if p in lower_filename), "")
        
        # Construct new filename in the correct order
        new_filename_parts = []
        if model_name:
            new_filename_parts.append(model_name)
        if dataset_name:
            new_filename_parts.append(dataset_name)
        new_filename_parts.append(train_portion)  # Ensure train_portion comes before pretrain status
        if pretrained:
            new_filename_parts.append(pretrained)
        
        new_filename = "_".join(new_filename_parts) + ".pkl"
        
        # Rename file if different from the original
        old_filepath = os.path.join(directory, filename)
        new_filepath = os.path.join(directory, new_filename)
        
        if old_filepath != new_filepath:
            print(f"Before: {filename}\nAfter: {new_filename}\n")
            os.rename(old_filepath, new_filepath)
        else:
            print(f"Skipped: {filename} (already follows naming convention)")
# Example usage
# rename_pickle_files("/Users/maxxyouu/Desktop/delong_test/delong_stats")
def extract_model_dataset(filename):
    """Extract model name and dataset name (if available) from the filename, avoiding 'trainPortion' as dataset."""
    match = re.match(r'(resnet|swin|densenet)(?:_([^_]+))?', filename)
    if match:
        model_name = match.group(1)
        dataset_name = match.group(2) if match.group(2) and not match.group(2).startswith("trainPortion") else None

        # Assign {model}_ours only for swin and resnet (not densenet)
        if dataset_name is None and model_name in {"swin", "resnet"}:
            return f"{model_name}_ours"
        elif dataset_name is None and model_name == "densenet":
            return model_name  # Keep as "densenet", no "_ours"

        return f"{model_name}-{dataset_name}"  # Format as model-dataset
    return None  # Keep as None if pattern doesn't match

def extract_model_dataset(filename):
    """Extract model name and dataset name (if available) from the filename, avoiding 'trainPortion' as dataset."""
    match = re.match(r'(resnet|swin|densenet)(?:_([^_]+))?', filename)
    if match:
        model_name = match.group(1)
        dataset_name = match.group(2) if match.group(2) and not match.group(2).startswith("trainPortion") else None

        # Assign {model}_ours only for swin and resnet (not densenet)
        if dataset_name is None and model_name in {"swin", "resnet"}:
            return f"{model_name}_ours"
        elif dataset_name is None and model_name == "densenet":
            return model_name  # Keep as "densenet", no "_ours"

        return f"{model_name}-{dataset_name}"  # Format as model-dataset
    return None  # Keep as None if pattern doesn't match

def load_pickle_objects(directory, include_trainPortion1=False):
    """Load pickle files, grouping by (model, dataset), and filtering out pretrainTrue.
    Optionally include trainPortion1 based on `include_trainPortion1` flag.
    """
    pickle_data = defaultdict(list)

    for filename in os.listdir(directory):
        if not filename.endswith('.pkl'):
            continue

        # Skip files that contain "pretrainTrue"
        if "pretrainTrue" in filename:
            continue  

        model_dataset = extract_model_dataset(filename)
        if not model_dataset:
            model_dataset = filename  # If extraction fails, use filename as group key

        # Extract trainPortion number (handles both integers and floats)
        train_match = re.search(r'trainPortion(\d*\.?\d+)', filename)
        if train_match:
            train_portion = float(train_match.group(1))  # Convert to float
            
            # Conditionally skip trainPortion1 based on the toggle
            if not include_trainPortion1 and train_portion == 1:
                continue 

            file_path = os.path.join(directory, filename)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)  # Load pickle object
            
            pickle_data[model_dataset].append((train_portion, data))  # Group by model-dataset

    return pickle_data

def calculate_auc_score(labels, pred_probs, metric="roc"):
    """Compute either ROC AUC or PR AUC (Precision-Recall) based on the `metric` argument."""
    if metric == "roc":
        return roc_auc_score(labels, pred_probs, average='micro', multi_class='ovr')
    elif metric == "pr":
        return average_precision_score(labels, pred_probs, average='micro')
    else:
        raise ValueError("Invalid metric. Choose 'roc' for ROC AUC or 'pr' for PR AUC.")

def process_and_plot(directory, include_trainPortion1=False, metric="roc"):
    """Process pickle files, group them by (model, dataset), calculate AUC scores (ROC or PR), and plot grouped lines."""
    pickle_data = load_pickle_objects(directory, include_trainPortion1)

    # Generate unique colors for each model-dataset pair
    num_groups = len(pickle_data)
    colors = cm.get_cmap('tab10', num_groups)

    # Plot results
    plt.figure(figsize=(8, 6))

    for idx, (model_dataset, files) in enumerate(pickle_data.items()):
        train_portions = []
        auc_scores = []

        for train_portion, data in files:
            labels = np.array(data['labels'])  # Ensure correct key
            pred_probs = np.array(data['pred_probs'])
            
            if labels.ndim == 1 or pred_probs.ndim == 1:  # Ensure correct shape
                labels = labels.reshape(-1, 1)
                pred_probs = pred_probs.reshape(-1, 1)

            try:
                auc_score = calculate_auc_score(labels, pred_probs, metric)
                train_portions.append(train_portion)
                auc_scores.append(auc_score)
            except ValueError as e:
                print(f"Skipping {model_dataset} (trainPortion {train_portion}): {e}")  # Handle errors

        # Sort values by trainPortion for smooth line connection
        sorted_indices = np.argsort(train_portions)
        sorted_train_portions = np.array(train_portions)[sorted_indices]
        sorted_auc_scores = np.array(auc_scores)[sorted_indices]

        plt.plot(sorted_train_portions, sorted_auc_scores, marker='o', linestyle='-', label=model_dataset, color=colors(idx))  # Connect dots

    # Final plot formatting
    plt.xlabel('Few Shot (in %)')
    plt.ylabel('PR AUC' if metric == "pr" else 'ROC AUC')
    plt.title(f'{"PR AUC" if metric == "pr" else "ROC AUC"} vs. Few-shots')
    plt.grid(True)
    plt.legend(loc='best', fontsize='small')  # Show legend with model-dataset pairs
    plt.show()
    print('done')

# Example usage
process_and_plot('/Users/maxxyouu/Desktop/delong_test/delong_stats', False, metric='pr')

# Example usage:
# To compute ROC AUC (default) and exclude trainPortion1:
# process_and_plot('/path/to/your/pickle/files')

# To compute ROC AUC and include trainPortion1:
# process_and_plot('/path/to/your/pickle/files', include_trainPortion1=True)

# To compute PR AUC and exclude trainPortion1:
# process_and_plot('/path/to/your/pickle/files', metric="pr")

# To compute PR AUC and include trainPortion1:
# process_and_plot('/path/to/your/pickle/files', include_trainPortion1=True, metric="pr")