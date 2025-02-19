import os
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pydicom
from sklearn.model_selection import train_test_split

# =======================
# MODEL DEFINITION
# =======================

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, image_size):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, in_channels=1, image_size=512, patch_size=16, embed_dim=768, 
                 num_heads=4, depth=6, num_classes=2, mlp_ratio=4.0, dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim, image_size)
        num_patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout) 
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_token_final = x[:, 0]
        out = self.head(cls_token_final)
        return out

# =======================
# PART 1: NIFTI Data Loading & Train/Test Split
# =======================

# File paths for the CT volume and infection mask (NIFTI files)
ct_volume_path = r'C:\Users\BAPS\Documents\Dicom Analaysis\Dicom_Analyzer\Slice_Classifier\Dataset\Covid_Dataset\ct_scans\coronacases_org_001.nii'
infection_mask_path = r'C:\Users\BAPS\Documents\Dicom Analaysis\Dicom_Analyzer\Slice_Classifier\Dataset\Covid_Dataset\infection_mask\coronacases_001.nii'

# Load NIFTI files using nibabel
ct_volume = nib.load(ct_volume_path).get_fdata()          # Expected shape: (512, 512, num_slices)
infection_mask = nib.load(infection_mask_path).get_fdata()  # Should match ct_volume shape

assert ct_volume.shape == infection_mask.shape, "CT volume and infection mask shapes do not match!"

# Create labels: 1 if any infection is present in the slice, else 0
abnormal_slices = [1 if np.any(infection_mask[:, :, idx]) else 0 
                   for idx in range(infection_mask.shape[2])]

# DataFrame with slice numbers and labels
df = pd.DataFrame({
    'slice_number': list(range(infection_mask.shape[2])),
    'label': abnormal_slices
})

print(f"Total slices: {len(df)}")
print(f"Abnormal slices: {df['label'].sum()} ({(df['label'].sum() / len(df)) * 100:.2f}%)")

# Split indices into train and test sets (stratified)
X = df['slice_number'].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Train slices: {len(X_train)}")
print(f"Test slices: {len(X_test)}")

# =======================
# PART 2: Load Pretrained Model
# =======================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VisionTransformer(
    in_channels=1,
    image_size=512,
    patch_size=16,
    embed_dim=768,
    num_heads=4,
    depth=6,
    num_classes=2,
    mlp_ratio=4.0,
    dropout=0.1
).to(device)

model_path = r'Vision Transformer\best_model.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# =======================
# PART 3: Inference on DICOM Slices
# =======================

# Path to folder containing DICOM slices (and subfolders)
dicom_root_folder = r"C:\Users\BAPS\Documents\Dicom Analaysis\Dicom_Dataset\Single Sliced Dataset\CMB-MML\MSB-00140"

# Dataset mean and std from training (update if necessary)
dataset_mean = 0.5
dataset_std = 0.25

# List to store detected anomalies
anomalies = []

# Recursively gather all file paths in the DICOM folder
all_file_paths = []
for dirpath, _, filenames in os.walk(dicom_root_folder):
    for file in filenames:
        filepath = os.path.join(dirpath, file)
        all_file_paths.append(filepath)

progress_bar = tqdm(all_file_paths, desc="Processing DICOM slices", ncols=100)

for file_path in progress_bar:
    try:
        dicom_data = pydicom.dcmread(file_path)
    except Exception:
        continue

    if not hasattr(dicom_data, 'pixel_array'):
        continue

    # Get pixel data, normalize and convert to tensor
    ct_slice = dicom_data.pixel_array.astype(np.float32)
    ct_slice_norm = (ct_slice - np.min(ct_slice)) / (np.ptp(ct_slice) + 1e-8)
    ct_tensor = torch.tensor(ct_slice_norm).unsqueeze(0)  # [1, H, W]
    
    # Resize to 512x512 (model expects 512x512)
    ct_tensor = ct_tensor.unsqueeze(0)  # [1, 1, H, W]
    ct_tensor = F.interpolate(ct_tensor, size=(512, 512), mode='bilinear', align_corners=False)
    ct_tensor = ct_tensor.squeeze(0)      # [1, 512, 512]
    
    # Standardize using training mean and std
    ct_tensor = (ct_tensor - dataset_mean) / dataset_std
    
    # Add batch dimension and move to device
    ct_tensor = ct_tensor.unsqueeze(0).to(device)  # [1, 1, 512, 512]
    
    # Run inference: get probabilities using softmax
    with torch.no_grad():
        output = model(ct_tensor)  # [1, 2]
        probs = torch.softmax(output, dim=1)
        anomaly_probability = probs[0, 1].item()  # Assuming class 1 indicates anomaly
    
    prediction = 1 if anomaly_probability > 0.5 else 0
    
    progress_bar.set_postfix_str(f"File: {os.path.basename(file_path)} | Pred: {prediction} | Prob: {anomaly_probability:.4f}")
    
    if prediction == 1:
        anomalies.append((file_path, anomaly_probability))

# =======================
# PART 4: Summary of Results
# =======================

print("\nAnomalies detected in the following slices:")
if anomalies:
    for path, prob in anomalies:
        print(f"{path} - Probability: {prob:.4f}")
else:
    print("No anomalies detected in the provided folder and its subfolders.")
