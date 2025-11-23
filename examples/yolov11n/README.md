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
├── export_compare_onnx.py     # Export and benchmark PyTorch vs ONNX models
├── compare_all_models.py      # Batch comparison across all pruning levels
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
pip install ultralytics onnx onnxruntime
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

### Model Comparison and Benchmarking

#### Single Model Comparison

Compare original and pruned models with PyTorch and ONNX benchmarking:
```bash
python export_compare_onnx.py \
  --orig runs/finetune_b16/weights/best.pt \
  --pruned runs/pruned_b16_75.0%_after_finetune/weights/best.pt \
  --imgsz 640 --batch 4 --iters 100 --simplify
```

This will:
- Export both models to ONNX format
- Benchmark PyTorch inference speed
- Benchmark ONNX inference speed
- Generate a JSON report with latency and speedup metrics

#### Batch Model Comparison

Compare all pruned models across multiple pruning levels automatically:
```bash
python compare_all_models.py \
  --imgsz 640 --batch 4 --iters 100 \
  --simplify --json --summary_md
```

**Optional Parameters:**
- `--limit_batches 16 8` - Restrict to specific batch sizes
- `--limit_prunes 20 40 50 75` - Restrict to specific pruning percentages
- `--runs_dir runs` - Directory containing finetune/pruned model subdirectories
- `--output_dir logs/compare_all` - Output directory for results
- `--summary_md` - Generate Markdown summary tables

**Output Files:**
- `logs/compare_all/benchmarks_img640_b4_<timestamp>.csv` - Aggregated CSV results
- `logs/compare_all/benchmarks_img640_b4_<timestamp>.json` - Detailed JSON metrics
- `logs/compare_all/summary_<timestamp>.md` - Markdown summary tables
- `logs/compare_all/run_<timestamp>.log` - Execution log with per-model details

**Logged Metrics:**
- Model parameters (millions)
- GFLOPs and FLOPs reduction percentage
- PyTorch inference latency and speedup
- ONNX inference latency and speedup
- ONNX execution provider (CPU/CUDA)

**Example Output:**
```
Baseline stats batch=16: params=2.590M, GFLOPs=3.16
Benchmarking batch=16 prune=20.0% ...
Done prune=20.0% in 2.8s | params=2.061M | GFLOPs=2.46 | FLOPs_Reduction=-22% | PT_speedup=1.09x | ONNX_speedup=1.11x
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