import os
import numpy as np
from tqdm import tqdm
import torch
import torchxrayvision as xrv
from PIL import Image
import SimpleITK as sitk
from scipy.linalg import sqrtm
from data_comparison import collect_one_mha_per_subfolder, collect_mha_files_flat, parse_mimic_jpg_files
import random

# --- 1. Load CheXNet from torchxrayvision ---
def load_chexnet_model():
    model = xrv.models.DenseNet(
        weights="densenet121-res224-all", 
        cache_dir='/cluster/projects/mcintoshgroup/CT-RATE-CHECKPOINTS/models/cheXnet/'
    )
    model.eval()
    return model.cuda() if torch.cuda.is_available() else model

# --- 2. Preprocess Image as Expected by CheXNet ---
def preprocess_image_xrv(path):
    """Load .mha medical image, center crop, normalize, and prepare for CheXNet input."""
    ext = os.path.splitext(path)[1].lower()

    # handle both file format case
    if ext in ['.jpg', '.jpeg', '.png']:
        img = Image.open(path).convert('L')  # Convert to grayscale
        arr = np.array(img).astype(np.float32)
    elif ext == '.mha':
        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img)
        
        # Take center slice if 3D
        if arr.ndim == 3:
            arr = arr[arr.shape[0] // 2]

    # Normalize to [0, 1] then scale to [-1, 1]
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    arr = (arr - 0.5) / 0.5  # Normalize to [-1, 1], since the previous line of code already normalize it to range [0, 1]

    # Center crop to 224x224
    h, w = arr.shape
    ch, cw = 224, 224
    start_h = max((h - ch) // 2, 0)
    start_w = max((w - cw) // 2, 0)
    cropped = arr[start_h:start_h + ch, start_w:start_w + cw]

    # Pad if needed
    padded = np.zeros((ch, cw), dtype=np.float32)
    padded[:cropped.shape[0], :cropped.shape[1]] = cropped

    # Repeat grayscale channel to 3 channels for CheXNet
    img_tensor = torch.tensor(padded).unsqueeze(0)  # (1, H, W)
    return img_tensor.unsqueeze(0)  # (1, 3, H, W)

# --- 3. Extract Features for a Folder ---
def extract_chexnet_features(folder, model, sample_size=100):
    feats = []
    if 'mimic' not in folder.lower(): # default is CT-RATE dataset
        files = collect_one_mha_per_subfolder(folder)
        files = random.sample(files, sample_size)
    elif 'MIMIC-CXR-JPG' in folder: # the whole MIMIC-CXR-JPG folder
        files = parse_mimic_jpg_files(
            mimic_jpg_label_csv='/cluster/projects/mcintoshgroup/publicData/CT-RATE/preprocessed_mimic/mimic-cxr-2.0.0-metadata.csv',
            root_dir=folder,
            sample_size=sample_size
        )
    else: # the MIMIC-CT folder
        files = collect_mha_files_flat(folder)
        files = random.sample(files, sample_size)

    for fname in tqdm(sorted(files), desc=f"Extracting from {os.path.basename(folder)}"):
        path = os.path.join(folder, fname)
        if not os.path.isfile(path): continue
        try:
            img = preprocess_image_xrv(path)
            img = img.cuda() if torch.cuda.is_available() else img
            with torch.no_grad():
                feat = model.features2(img).squeeze().cpu().numpy()
            feats.append(feat)
        except Exception as e:
            print(f"Error processing {fname}: {e}")
    return np.array(feats)

# --- 4. FID Computation ---
def compute_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)

# --- 5. Wrapper Function ---
def compute_medical_fid_chexnet(folder1, folder2, sample_size):
    model = load_chexnet_model()
    feats1 = extract_chexnet_features(folder1, model, sample_size)
    feats2 = extract_chexnet_features(folder2, model, sample_size)

    mu1, sigma1 = np.mean(feats1, axis=0), np.cov(feats1, rowvar=False)
    mu2, sigma2 = np.mean(feats2, axis=0), np.cov(feats2, rowvar=False)

    fid_score = compute_fid(mu1, sigma1, mu2, sigma2)
    print(f"\n✅ Medical FID (CheXNet features, ChestX-ray14): {fid_score}")
    return fid_score

# --- Run Example ---
if __name__ == "__main__":
    folder1 = '/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/train_preprocessed_xray_mha'  # Contains nested folders
    # folder2 = '/cluster/projects/mcintoshgroup/publicData/CT-RATE/preprocessed_mimic/mimic_preprocessed_xray_mha'  # Flat folder with .mha files
    folder2 = '/cluster/projects/mcintoshgroup/publicData/MIMIC-CXR/MIMIC-CXR-JPG'
    compute_medical_fid_chexnet(folder1, folder2, sample_size=200)

    # conclusion, with the 200 images, they are indistinguishable.

    # TODO: base on https://github.com/bioinf-jku/TTUR we should try bigger size (use the whole mimic-jpg)
    # mimic_jpg_files = parse_mimic_jpg_files(
    #         mimic_jpg_label_csv='/cluster/projects/mcintoshgroup/publicData/CT-RATE/preprocessed_mimic/mimic-cxr-2.0.0-metadata.csv',
    #         root_dir='/cluster/projects/mcintoshgroup/publicData/MIMIC-CXR/MIMIC-CXR-JPG',
    #         sample_size=5
    #     )
    # print('done')

    # the whole mimic-cxr-jpg dataset label file to get the frontal view
    # /cluster/projects/mcintoshgroup/publicData/CT-RATE/preprocessed_mimic/mimic-cxr-2.0.0-metadata.csv
    # the directory that list all the files.
    # /cluster/projects/mcintoshgroup/publicData/MIMIC-CXR/MIMIC-CXR-JPG