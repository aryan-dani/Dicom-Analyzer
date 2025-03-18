import pydicom
import pywt
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms, models
import urllib.request

def dicom_to_wavelet_image(dicom_path, wavelet='haar'):
    """
    Reads a DICOM file, normalizes the image, extracts the middle slice if 3D,
    applies a 2D Discrete Wavelet Transform (DWT), and returns the normalized
    approximation (LL) coefficients as an 8-bit image.
    """
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array.astype(np.float32)
    
    # Normalize image to [0, 1]
    img_norm = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
    
    # If 3D, select the middle slice
    if img_norm.ndim == 3:
        img_norm = img_norm[img_norm.shape[0] // 2]
    
    # Apply 2D DWT
    coeffs2 = pywt.dwt2(img_norm, wavelet)
    LL, (LH, HL, HH) = coeffs2  # LL holds the low-frequency content
    
    # Normalize LL to [0, 255] and convert to uint8
    LL_scaled = (LL - np.min(LL)) / (np.max(LL) - np.min(LL) + 1e-8)
    LL_scaled = (LL_scaled * 255).astype(np.uint8)
    return LL_scaled

def main():
    # Update this path to your DICOM file
    dicom_path = "path/to/your/dicom_file.dcm"
    
    # Apply wavelet transform to the DICOM image
    wave_img = dicom_to_wavelet_image(dicom_path, wavelet='haar')
    cv2.imwrite("wavelet_output.png", wave_img)
    print("Wavelet transformed image saved as 'wavelet_output.png'.")
    
    # Convert the single-channel image to a 3-channel image for the classifier
    wave_img_3ch = cv2.cvtColor(wave_img, cv2.COLOR_GRAY2BGR)
    pil_img = Image.fromarray(wave_img_3ch)
    
    # Preprocess the image for ResNet-18 (pretrained on ImageNet)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(pil_img)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model
    
    # Load a pre-trained ResNet-18 classifier
    model = models.resnet18(pretrained=True)
    model.eval()
    
    # Run inference
    with torch.no_grad():
        output = model(input_batch)
    _, predicted = torch.max(output, 1)
    predicted_index = predicted.item()
    print("Predicted class index:", predicted_index)
    
    # Download ImageNet class labels
    url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
    with urllib.request.urlopen(url) as f:
        labels = [line.decode('utf-8').strip() for line in f.readlines()]
    
    # Print the human-readable label
    if predicted_index < len(labels):
        predicted_label = labels[predicted_index]
        print("Predicted label:", predicted_label)
    else:
        print("Predicted index out of range for ImageNet labels.")

if __name__ == "__main__":
    main()
