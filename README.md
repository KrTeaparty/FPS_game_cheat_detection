# FPS Game Cheat Detection

An implementation of a Video-based Anomaly (Cheat/Bot) Detection model designed for First-Person Shooter (FPS) games, using human behavioral features extracted via 3D Convolutional Neural Networks (I3D).

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red)

## Overview
This repository provides the core training and testing code for detecting cheats or bots in FPS games like Counter-Strike 2.
1. **Feature Extraction**: Extracting 1024-dimensional clip-level spatiotemporal features from raw match videos using an I3D model. We compare non-fine-tuned (`ft0`) and fine-tuned (`ft1`) variations.
2. **Anomaly Detection (5-Fold Cross Validation)**: Training an anomaly detection network using the extracted features. A Match-level GroupKFold strategy is applied so that training and validation videos from the same match do not overlap.

## Repository Structure
```
├── gkf_splits/                              # Pickled index files containing Match-level Fold splits
├── list/                                    # Text files containing data paths for data loaders
├── models/                                  # Trained 5-Fold UR-DMU models
├── outputs/                                 # Evaluation results (Precision, Recall, F1, Accuracy)
├── synthetic_results/                       # ROC-AUC graphs and subplots for Test videos
├── synthetic_test_data/                     # Raw ground-truth arrays and .mp4s for Synthetic Testing
├── compressed_features/                     # Directory for split zip archives (Raw features)
├── auto_train_5fold.py                      # Main script: GroupKFold 5-Fold CV Training (RAM Cached)
├── generate_lists_and_gkf.py                # Script to generate Match-level splits
├── i3d_finetuning.py                        # I3D Fine-tuning script
├── feature_extraction_v2.py                 # Feature extraction script with frozen FC weights
├── create_synthetic_test.py / eval_synthetic.py # Scripts for creating and evaluating synthetic test videos
├── dataset_loader.py                        # Dataloader with high-performance RAM caching
├── model.py                                 # Core Cheat Detection Network Architecture
├── best_i3d_model_v4.pth                    # Fine-tuned I3D backbone
├── fc_weights.pth                           # Seeded Random Projection weights
└── requirements.txt                         # Package dependencies
```

## Dataset & Models Preparation

1. **Features & Maps (`compressed_features/cs2_feat_8_ft0.zip`, `list/`)**
   We have included the extracted video features `.npy` and map lists (strides 8, 16) in this repository so you can reproduce the results immediately.
   Due to GitHub's directory file limits, all features (`cs2_feat_*`) have been bundled into a single split zip archive placed inside the `compressed_features` directory. Please extract this multi-part archive into the project root directory before running the code.
2. **Pre-trained Weights & Models (`models/`, `pth`)**
   The evaluations for strides 8 and 16 under both fine-tuning setups are provided in the `models/` directory. You can use the included `best_i3d_model_v4.pth` backbone for any new video extractions.

## Usage

### 1. Environments
Please ensure you have Python 3.10+ and an environment capable of running PyTorch 2.5+. You can install the required packages using the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

### 2. Training & Evaluation (Main Pipeline)
The entire training and testing process is automated through Match-level 5-Fold Cross Validation.

```bash
python auto_train_5fold.py
```
This script will automatically:
* Load data using the optimized RAM-caching `dataset_loader.py`.
* Read Match-level split indices from `gkf_splits/gkf_5fold_idx.pickle`.
* Iterate over different strides (8, 16) and configurations (`ft0`, `ft1`).
* Output `Accuracy`, `Precision`, `Recall`, and `F1-Score` to the `outputs/` directory.

### 3. Feature Extraction
To fine-tune the model and extract features from your own game videos, use:
```bash
python i3d_finetuning.py
python feature_extraction_v2.py
```

### 4. Synthetic Video Evaluation (ROC-AUC)
To reconstruct the time-series anomaly detection evaluation and plot graphs:
```bash
python eval_synthetic.py
```
Outputs are saved as high-resolution images in `synthetic_results/`.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
