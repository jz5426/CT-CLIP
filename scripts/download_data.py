from huggingface_hub import hf_hub_download, list_repo_files
import os
import shutil
from tqdm import tqdm

repo_id = "ibrahimhamamci/CT-RATE"
folder_path = "dataset/valid"

# List all files in the repository
all_files = list_repo_files(repo_id, repo_type="dataset")

# Filter files in the 'valid' folder
valid_files = [f for f in all_files if f.startswith(folder_path)]

print(f"Files in the 'valid' folder: {valid_files}")

destination_folder = 'F:\\Chris\\CT-RATE'
os.makedirs(destination_folder, exist_ok=True)

for file in tqdm(valid_files[:1000]):
    # Extract the filename from the path
    filename = os.path.basename(file)
    dir_path = os.path.dirname(file)

    # Download the file
    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=dir_path,
        repo_type="dataset",
        token='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' # Ensure your token is set in the huggingface
    )

    # Normalize each component to use consistent separators
    destination_folder = os.path.normpath(destination_folder)
    dir_path = os.path.normpath(dir_path)

    # Ensure the destination directory exists
    destination_dir = os.path.join(destination_folder, dir_path)
    os.makedirs(destination_dir, exist_ok=True)

    # Move the downloaded file to the destination folder
    file_dir = os.path.join(destination_dir, filename)
    shutil.move(file_path, file_dir)

print(f"Downloaded files are saved in {destination_folder}")
