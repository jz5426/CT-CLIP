import os
import numpy as np
from tqdm import tqdm
# from PIL import Image
import torch
# import torchvision.transforms as transforms
import torchxrayvision as xrv
import SimpleITK as sitk
from scipy.linalg import sqrtm
from data_comparison import collect_one_mha_per_subfolder, collect_mha_files_flat

# --- 1. Load CheXNet from torchxrayvision ---
def load_chexnet_model():
    model = xrv.models.DenseNet(weights="densenet121-res224-all", cache_dir='')
    model.eval()
    return model.cuda() if torch.cuda.is_available() else model

# --- 2. Preprocess Image as Expected by CheXNet ---
def preprocess_image_xrv(path):
    # TODO:
    # img = Image.open(path).convert('L')  # single-channel grayscale
    # img = img.resize((224, 224))
    # img = np.array(img).astype(np.float32) / 255.0
    # img = (img - 0.5) / 0.5  # Normalize to [-1, 1]
    # img = np.expand_dims(img, axis=0)  # (1, H, W)
    # img = np.repeat(img, 3, axis=0)  # (3, H, W)
    # img_tensor = torch.tensor(img).unsqueeze(0)  # (1, 3, H, W)
    # return img_tensor

    """Load .mha medical image, center crop, normalize, and prepare for CheXNet input."""
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
    img_tensor = img_tensor.repeat(3, 1, 1)  # (3, H, W)
    return img_tensor.unsqueeze(0)  # (1, 3, H, W)

# --- 3. Extract Features for a Folder ---
def extract_chexnet_features(folder, model):
    feats = []

    if 'mimic' not in folder:
        files = collect_one_mha_per_subfolder(folder)
    else:
        files = collect_mha_files_flat(folder)

    for fname in tqdm(sorted(files), desc=f"Extracting from {os.path.basename(folder)}"):
        path = os.path.join(folder, fname)
        if not os.path.isfile(path): continue
        try:
            img = preprocess_image_xrv(path)
            img = img.cuda() if torch.cuda.is_available() else img
            with torch.no_grad():
                feat = model.features(img).squeeze().cpu().numpy()
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
def compute_medical_fid_chexnet(folder1, folder2):
    model = load_chexnet_model()
    feats1 = extract_chexnet_features(folder1, model)
    feats2 = extract_chexnet_features(folder2, model)

    mu1, sigma1 = np.mean(feats1, axis=0), np.cov(feats1, rowvar=False)
    mu2, sigma2 = np.mean(feats2, axis=0), np.cov(feats2, rowvar=False)

    fid_score = compute_fid(mu1, sigma1, mu2, sigma2)
    print(f"\n✅ Medical FID (CheXNet features, ChestX-ray14): {fid_score:.4f}")
    return fid_score

# --- Run Example ---
if __name__ == "__main__":
    folder1 = '/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/valid_preprocessed_xray_mha'  # Contains nested folders
    folder2 = '/cluster/projects/mcintoshgroup/publicData/CT-RATE/preprocessed_mimic/mimic_preprocessed_xray_mha'  # Flat folder with .mha files
    compute_medical_fid_chexnet(folder1, folder2)
