# YOLO-based Model Pruning and Fine-tuning Research

## Overview

This research project focuses on pruning and fine-tuning YOLO models (specifically YOLOv11) for object detection tasks. The project implements various pruning strategies, custom architectures (C3K2), and comprehensive training pipelines for model optimization.

## Project Structure

```
├── c3k2_v2.py                 # Custom C3K2 architecture implementation
├── finetune.py                # Fine-tuning scripts for YOLO models
├── prune.py                   # Main pruning implementation
├── single_prune.py            # Single-layer pruning utilities
├── yolov11n_pruning.py        # YOLOv11n-specific pruning pipeline
├── train_v2.py                # Training pipeline (version 2)
├── metrics.py                 # Evaluation metrics and performance tracking
├── yolo11n.pt                 # Pre-trained YOLOv11n weights
├── yolo11n_c3k2.txt          # C3K2 architecture configuration
├── run.sh                     # Main execution script
├── exp_1.sh                   # Experiment 1 execution script
├── BCCD/                      # Blood Cell Count and Detection dataset
├── examples/                  # Example implementations and outputs
├── logs/                      # Training and experiment logs
└── runs/                      # Training runs and results
```

## Dataset

### TXL-PBC Dataset

This project uses the TXL-PBC (Peripheral Blood Cell) dataset for training and evaluation.

**Download the dataset from:**
https://figshare.com/articles/dataset/TXL-PBC_Dataset/27073186/8

After downloading, extract the dataset to the `27073186/TXL-PBC/` directory in the project root.

### BCCD Dataset

The Blood Cell Count and Detection (BCCD) dataset is also included for additional experiments. The dataset structure follows the YOLO format with separate `train/`, `valid/`, and `test/` splits.

## Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <project-directory>
```

2. Create and activate virtual environment:
```bash
python -m venv ultravenv
source ultravenv/bin/activate  # On Windows: ultravenv\Scripts\activate
```

3. Install dependencies:
```bash
pip install ultralytics
```

4. Download the TXL-PBC dataset from the link above and place it in the appropriate directory.

## Usage

### Training

Run the main training script:
```bash
bash run.sh
```

Or use the training module directly:
```bash
python train_v2.py
```

### Pruning

Execute model pruning:
```bash
python prune.py
```

For YOLOv11n-specific pruning:
```bash
python yolov11n_pruning.py
```

### Fine-tuning

Fine-tune a pruned or pre-trained model:
```bash
python finetune.py
```

### Running Experiments

Execute predefined experiments:
```bash
bash exp_1.sh
```

## Evaluation

Metrics and evaluation utilities are provided in `metrics.py`. Results are automatically saved to the `runs/` directory.

## Results

Training runs, detection results, and fine-tuned models are stored in:
- `runs/detect/` - Detection results
- `runs/fine_tuned_yolo11n/` - Fine-tuned model outputs
- `logs/` - Training logs and metrics

## Dataset
@dataset{txl_pbc_2024,
  title={TXL-PBC Dataset},
  author={Gan, Lu; Li, Xi; Wang, Xichun},
  year={2024},
  publisher={figshare},
  url={https://figshare.com/articles/dataset/TXL-PBC_Dataset/27073186/8}
}
```