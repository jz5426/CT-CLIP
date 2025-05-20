"""
This script mainly to generate histogram EMD to compare synthetic and real xray data
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import SimpleITK as sitk
from scipy.stats import wasserstein_distance
import random
import pandas as pd
import pickle

# --- Helper Functions ---

def parse_mimic_jpg_labels(mimic_jpg_label_csv):
    # Load the CSV file
    df = pd.read_csv(mimic_jpg_label_csv)

    # Filter rows where ViewCodeSequence_CodeMeaning is "antero-posterior"
    filtered_df = df[df['ViewCodeSequence_CodeMeaning'].str.lower() == 'antero-posterior']

    # Create the dictionary
    dicom_to_view = dict(zip(filtered_df['dicom_id'], filtered_df['ViewCodeSequence_CodeMeaning']))

    # Optional: print or inspect
    print(f"Created dictionary with {len(dicom_to_view)} entries")
    return set(dicom_to_view.keys())

def parse_mimic_jpg_files(mimic_jpg_label_csv, root_dir, sample_size):

    # retrieve the object file if exists
    pickle_path = os.path.join(mimic_jpg_label_csv, 'an_mimic_jpg.pkl')
    if os.path.exists(pickle_path):
        print(f"Loading matched paths from existing pickle: {pickle_path}")
        with open(pickle_path, "rb") as f:
            matched_jpg_paths = pickle.load(f)
        matched_jpg_paths = random.sample(matched_jpg_paths, sample_size)
        return matched_jpg_paths

    dicom_to_view = parse_mimic_jpg_labels(mimic_jpg_label_csv)
    matched_jpg_paths = []

    # Traverse all subdirectories
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith('.jpg'):
                dicom_id = os.path.splitext(fname)[0]
                if dicom_id in dicom_to_view:
                    full_path = os.path.join(dirpath, fname)
                    matched_jpg_paths.append(full_path)

    print(f"Found {len(matched_jpg_paths)} matching .jpg files.")

    # Save the list to the pickle file so that next time no need to parse it again
    with open(pickle_path, "wb") as f:
        pickle.dump(matched_jpg_paths, f)
    matched_jpg_paths = random.sample(matched_jpg_paths, sample_size)
    return matched_jpg_paths

def load_grayscale_image(path):
    """Load .mha medical image and convert to 2D grayscale numpy array."""
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # shape: (depth, height, width)
    if arr.ndim == 3: # if it is 3d.
        arr = arr[arr.shape[0] // 2]  # Take center slice

    # normalize the input range to make sense. this is what the model sees
    arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
    h, w = arr.shape
    ch, cw = 224, 224
    start_h = max((h - ch) // 2, 0)
    start_w = max((w - cw) // 2, 0)
    cropped = arr[start_h:start_h + ch, start_w:start_w + cw]

    # If the crop goes out of bounds, pad it
    padded = np.zeros((ch, cw), dtype=arr.dtype)
    padded[:cropped.shape[0], :cropped.shape[1]] = cropped

    return padded.astype(np.float32)

def compute_histogram(img, bins=256):
    """Compute normalized histogram of an image."""
    hist, _ = np.histogram(img.flatten(), bins=bins, range=(0, 255), density=True)
    return hist

def collect_one_mha_per_subfolder(folder):
    """Collect at most one .mha file per subdirectory."""
    selected_files = []
    for root, _, files in os.walk(folder):
        mha_files = [f for f in files if f.endswith('.mha')]
        if mha_files:
            chosen = random.choice(mha_files)
            selected_files.append(os.path.join(root, chosen))
    return selected_files

def collect_mha_files_flat(folder):
    """Collect all .mha files directly in the folder (non-recursive)."""
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.mha')]

# --- Main Computation ---
def compute_histogram_emd_and_plot(folder1, folder2, n=50, bins=256):
    assert 'mimic' in folder2
    files1 = collect_one_mha_per_subfolder(folder1)
    files2 = collect_mha_files_flat(folder2)

    if len(files1) < n or len(files2) < n:
        raise ValueError("Not enough .mha files in one of the folders to sample the requested amount.")

    sampled1 = random.sample(files1, n)
    sampled2 = random.sample(files2, n)

    emd_scores = []
    hist1_all = []
    hist2_all = []

    for f1, f2 in tqdm(zip(sampled1, sampled2), total=n, desc="Computing Histogram EMD"):
        img2 = load_grayscale_image(f2) # mimic version
        img1 = load_grayscale_image(f1) # synthetic version
        
        hist1 = compute_histogram(img1, bins)
        hist2 = compute_histogram(img2, bins)

        hist1_all.append(hist1)
        hist2_all.append(hist2)

        emd = wasserstein_distance(hist1, hist2)
        emd_scores.append(emd)

    mean_emd = np.mean(emd_scores)
    print(f"\nMean Histogram EMD between {folder1} and {folder2}: {mean_emd:.6f}")

    # Compute average histograms
    avg_hist1 = np.mean(hist1_all, axis=0)
    avg_hist2 = np.mean(hist2_all, axis=0)

    # Plotting
    plt.figure(figsize=(10, 6))
    bin_edges = np.linspace(0, 255, bins)
    plt.plot(bin_edges, avg_hist1, label='Synthetic CXRs from CT-RATE', lw=2)
    plt.plot(bin_edges, avg_hist2, label='MIMIC CXRs', lw=2)
    plt.fill_between(bin_edges, avg_hist1, alpha=0.3)
    plt.fill_between(bin_edges, avg_hist2, alpha=0.3)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Normalized Frequency')
    plt.ylim(0, 0.1)  # <- set Y-axis range here
    plt.title(f'Average Histogram Comparison\nMean EMD = {mean_emd:.6f}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'/cluster/home/t135419uhn/CT-CLIP/data_postprocess/histogram_emd_sample_size_{n}_bins_{bins}.png')

    # return mean_emd

# --- Main Computation t times---
# def compute_histogram_emd_t_times_and_plot(folder1, folder2, n=50, t=5, bins=256):
#     files1_all = collect_one_mha_per_subfolder(folder1)
#     files2_all = collect_mha_files_flat(folder2)

#     if len(files1_all) < n or len(files2_all) < n:
#         raise ValueError("Not enough .mha files in one of the folders to sample the requested amount.")

#     all_trial_emd = []

#     for trial in range(t):
#         sampled1 = random.sample(files1_all, n)
#         sampled2 = random.sample(files2_all, n)

#         emd_scores = []
#         hist1_all = []
#         hist2_all = []

#         for f1, f2 in tqdm(zip(sampled1, sampled2), total=n, desc=f"Trial {trial + 1}/{t}"):
#             img2 = load_grayscale_image(f2)
#             img1 = load_grayscale_image(f1)

#             hist1 = compute_histogram(img1, bins)
#             hist2 = compute_histogram(img2, bins)

#             hist1_all.append(hist1)
#             hist2_all.append(hist2)

#             emd = wasserstein_distance(hist1, hist2)
#             emd_scores.append(emd)

#         trial_mean_emd = np.mean(emd_scores)
#         all_trial_emd.append(trial_mean_emd)
#         print(f"Trial {trial + 1} mean EMD: {trial_mean_emd:.6f}")

#     final_mean = np.mean(all_trial_emd)
#     final_std = np.std(all_trial_emd)

#     print(f"\n🔁 Repeated {t} times:")
#     print(f"✅ Final Mean EMD: {final_mean:.6f}")
#     print(f"📏 Standard Deviation: {final_std:.6f}")

#     # Plot histogram from last trial
#     avg_hist1 = np.mean(hist1_all, axis=0)
#     avg_hist2 = np.mean(hist2_all, axis=0)

#     plt.figure(figsize=(10, 6))
#     bin_edges = np.linspace(0, 255, bins)
#     plt.plot(bin_edges, avg_hist1, label=f'{os.path.basename(folder1)}', lw=2)
#     plt.plot(bin_edges, avg_hist2, label=f'{os.path.basename(folder2)}', lw=2)
#     plt.fill_between(bin_edges, avg_hist1, alpha=0.3)
#     plt.fill_between(bin_edges, avg_hist2, alpha=0.3)
#     plt.xlabel('Pixel Intensity')
#     plt.ylabel('Normalized Frequency')
#     plt.title(f'Average Histogram (Last Trial)\nMean EMD = {trial_mean_emd:.6f}')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
    # return final_mean, final_std

# --- Run ---
if __name__ == "__main__":

    # one run to get the EMD comparison
    folder1 = '/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/valid_preprocessed_xray_mha'  # Contains nested folders
    folder2 = '/cluster/projects/mcintoshgroup/publicData/CT-RATE/preprocessed_mimic/mimic_preprocessed_xray_mha'  # Flat folder with .mha files
    sample_size = 200  # Adjust as needed
    compute_histogram_emd_and_plot(folder1, folder2, n=sample_size, bins=128)

    # multiple runs to ge thte EMD comparison
    # folder1 = '/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/valid_preprocessed_xray_mha'
    # folder2 = '/cluster/projects/mcintoshgroup/publicData/CT-RATE/preprocessed_mimic/mimic_preprocessed_xray_mha'
    # sample_size = 100
    # num_trials = 5

    # compute_histogram_emd_t_times_and_plot(folder1, folder2, n=sample_size, t=num_trials)
