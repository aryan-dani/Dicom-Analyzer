# Medical Image Classification Explorations

This repository explores various deep learning models and techniques for medical image analysis. It focuses on working with common medical imaging formats like DICOM and NIfTI, and tackling classification tasks such as tumor detection and infection localization.

## Repository Overview

This repository is organized into several top-level directories, each representing a distinct experiment or model approach for medical image classification. These directories contain the necessary code, scripts, and potentially model-specific documentation related to their respective approaches. The `Documentation` directory provides additional details that may span across multiple experiments or offer deeper insights into specific model training processes.

## Getting Started

To explore the experiments within this repository:
1.  **Navigate to a project directory:** Each top-level directory (e.g., `Classifier`, `InceptionNet`) contains a specific experiment.
2.  **Update dataset paths:** Notebooks and scripts within these directories will likely contain placeholders or example paths for datasets. You will need to update these paths to point to your local data.
3.  **Check dependencies:** While common libraries like TensorFlow, PyTorch, Pydicom, and Nibabel are used, specific scripts or notebooks might have additional dependencies. Refer to individual files or any `requirements.txt` files within project directories for details. Installation of these dependencies (e.g., using `pip install <package_name>`) might be required.

## Directory Descriptions

### Classifier
*   **Goal:** Brain Tumor Classification with CNN.
*   **Tools/Libraries:** TensorFlow/Keras, Pydicom, Nibabel.
*   **Input Data Format:** DICOM.
*   **Model Architecture:** Custom CNN.
*   **Scripts:** Includes training, evaluation, and inference scripts.

### InceptionNet
*   **Goal:** COVID-19 Infection Classification.
*   **Tools/Libraries:** PyTorch, Timm, Nibabel.
*   **Input Data Format:** NIfTI.
*   **Model Architecture:** InceptionResNetV2.
*   **Scripts:** Includes training and evaluation scripts/notebooks.

### Slice_Classifier
*   **Goal:** COVID-19 Infection Classification using NIfTI for training and also supports DICOM for inference.
*   **Tools/Libraries:** PyTorch, Nibabel, Pydicom.
*   **Input Data Format:** NIfTI (training), DICOM (inference).
*   **Model Architecture:** Custom CNN.
*   **Scripts:** Includes training scripts and inference notebooks/scripts.

### Vision Transformer
*   **Goal:** Likely COVID-19 Infection Classification.
*   **Tools/Libraries:** PyTorch, Timm, Nibabel, Pydicom.
*   **Input Data Format:** NIfTI (training notebook), DICOM (inference notebook and .py script).
*   **Model Architecture:** Vision Transformer (ViT).
*   **Scripts:** Includes training notebook, evaluation, and inference notebook/script.

### WaveTransform
*   **Goal:** Experiment using Wavelet Transforms as a preprocessing step for DICOM images, followed by classification using a pre-trained ResNet-18 on ImageNet classes.
*   **Tools/Libraries:** PyTorch, PyWavelets, Pydicom.
*   **Input Data Format:** DICOM.
*   **Model Architecture:** ResNet-18 with Wavelet features.
*   **Scripts:** `main.py` serves as an inference pipeline example; training/evaluation scripts may need development.

## Documentation

Further details on specific models and training processes can be found in the `Documentation` directory.
*   `Documentation/Classifier Model Training.pdf`: This document likely covers details about the training process, architecture, and performance of the image classifier for brain tumors located in the `Classifier` directory.

## Contributing

Contributions, suggestions, and issue reports are welcome. Please feel free to open an issue or submit a pull request.

## License

License information should be placed here. It is recommended to add a `LICENSE` file (e.g., `LICENSE.md` or `LICENSE.txt`) to the root of the repository detailing the terms under which the content is shared. For example: "This project is licensed under the MIT License - see the LICENSE.md file for details."
