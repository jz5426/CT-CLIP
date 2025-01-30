import os
import re

def rename_pickle_files(directory):
    """function to rename the file so that easiler to create a trend plots"""
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
rename_pickle_files("/Users/maxxyouu/Desktop/delong_test/delong_stats")
