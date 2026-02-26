# FPS Game Cheat Detection

An implementation of a Video-based Anomaly (Cheat/Bot) Detection model designed for First-Person Shooter (FPS) games, using human behavioral features extracted via 3D Convolutional Neural Networks (I3D).

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red)

## Overview
This repository provides the core training and testing code for detecting cheats or bots in FPS games like Counter-Strike 2. The pipeline is divided into two main stages:
1. **Feature Extraction**: Extracting 1024-dimensional clip-level spatiotemporal features from raw match videos using an I3D model. We compare non-fine-tuned (`ft0`) and fine-tuned (`ft1`) variations.
2. **Anomaly Detection (5-Fold Cross Validation)**: Training an anomaly detection network using the extracted features, separating normal player behavior from unnatural bot/cheat movements.

## Repository Structure
```
├── list/                                    # Text files containing data paths for data loaders
├── models/                                  # Pre-trained models and saved weights
├── compressed_features/                     # Directory for split zip archives to keep root clean
│   └── cs2_feat_8_ft0.zip (.z01~.z17)       # Split archive containing all extracted features & clip mappings
├── Sample_data/                             # Blurred/anonymized sample .mp4 videos
├── auto_train_5fold.py                      # Main script: 5-Fold Cross Validation Training & Evaluation
├── train.py / cs_test.py                    # Core training/testing loops and loss definitions
├── i3d_finetuning.py                        # I3D feature extraction script
├── model.py                                 # Core Cheat Detection Network Architecture Architecture
├── best_i3d_model_v3.zip                    # Compressed pre-trained I3D backbone weight used for feature extraction
└── requirements.txt                         # Package dependencies
```

> **Note:** Due to size limits, raw game video `.mp4` data (`raw_data/`) is not included in this repository.

## Dataset & Models Preparation

1. **Features & Maps (`compressed_features/cs2_feat_8_ft0.zip`, `list/`)**
   We have included the extracted video features `.npy` and map lists (strides 8, 16) in this repository so you can reproduce the results immediately.
   Due to GitHub's directory file limits and to keep the root directory clean, all features (`cs2_feat_*`) and clip match data (`data/Clip_match`) have been bundled into a single split zip archive named `cs2_feat_8_ft0` (`.zip`, `.z01` to `.z17`) placed inside the `compressed_features` directory. Please extract this multi-part archive into the project root directory before running the code.
   * Note on naming conventions: `ft0` indicates features extracted without fine-tuning, while `ft1` indicates features extracted with a fine-tuned model.
2. **Pre-trained Weights & Models (`models/`, `pth`)**
   The best performing evaluations for strides 8 and 16 under both fine-tuning setups are provided in the `models/` directory. If you wish to extract features yourself from new `.mp4` videos, you can use the included compressed `best_i3d_model_v3.zip` backbone (please extract it to obtain the `.pth` file before use).
3. **Sample Videos (`Sample_data/`)**
   A set of blurred/anonymized sample game clips (`.mp4`) showcasing Bot behavior is available in the `Sample_data` directory.

## Usage

### 1. Environments
Please ensure you have Python 3.10+ and an environment capable of running PyTorch 2.5+. You can install the required packages using the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

### 2. Training & Evaluation (Main Pipeline)
The entire training and testing process is automated through 5-Fold Cross Validation.

```bash
python auto_train_5fold.py
```
This script will automatically:
* Load data list from the `list/` directory.
* Iterate over different strides (8, 16) and fine-tuning configurations (`ft0` for no fine-tuning, `ft1` for fine-tuning).
* Perform 5-Fold evaluation and output `Accuracy`, `Precision`, `Recall`, and `F1-Score`.
* Save the best performing models into the `models/` directory.

### 3. Feature Extraction & AUC Validation
If you want to extract features from your own game videos:
1. Prepare game videos inside the respective class folders.
2. Run the extraction script (Update `dataset_root` in script manually if needed):
```bash
python i3d_finetuning.py
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
