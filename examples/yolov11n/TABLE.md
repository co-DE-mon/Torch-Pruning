# Experiment 1: YOLOv11n Pruning & Benchmark Summary

## Overview
This README consolidates pruning, benchmarking, and accuracy results for Experiment 1 using the TXL-PBC dataset. Models were fine-tuned and then pruned at multiple sparsity levels. Performance was measured in terms of parameters, FLOPs reduction, and PyTorch vs ONNX inference speedups.


## Consolidated Pruning & Benchmark Table
Collected across batch sizes (2, 4, 8, 16) using `compare_all_models.py` (imgsz=640, benchmarking batch=4). Speedups are relative to each batch size's finetuned baseline. mAP columns are baseline-only unless per-prune evaluation is performed (currently not collected for pruned checkpoints in this experiment).

| Batch | Prune % | Params (M) | FLOPs Red. | PyTorch Speedup | ONNX Speedup | mAP50 | mAP50-95 |
|-------|--------:|-----------:|-----------:|----------------:|-------------:|------:|---------:|
| 2 | 0 (baseline) | 2.59 | - | 1.00x | 1.00x | 0.983 | 0.850 |
| 2 | 5  | 2.43 | -8%  | 0.91x | 1.07x |  0.983  |   0.851  |
| 2 | 10 | 2.30 | -13% | 1.02x | 1.09x |  0.983  |   0.854  |
| 2 | 20 | 2.06 | -22% | 1.05x | 1.18x |  0.982  |   0.843  |
| 2 | 30 | 1.84 | -31% | 1.01x | 1.26x |  0.977  |   0.837  |
| 2 | 40 | 1.60 | -40% | 1.07x | 1.32x |  0.983  |   0.829  |
| 2 | 50 | 1.38 | -49% | 1.09x | 1.43x |  0.972  |   0.82   |
| 2 | 75 | 0.85 | -70% | 1.05x | 1.94x |  0.973  |   0.813  |
| 4 | 0 (baseline) | 2.59 | -    | 1.00x | 1.00x | 0.982 | 0.850 |
| 4 | 5  | 2.43 | -8%  | 0.93x | 1.03x |  0.986  |   0.864  |
| 4 | 10 | 2.30 | -13% | 1.01x | 1.13x |  0.984  |   0.862  |
| 4 | 20 | 2.06 | -22% | 1.05x | 1.17x |  0.985  |   0.864  |
| 4 | 30 | 1.84 | -31% | 1.03x | 1.29x |  0.981  |   0.858  |
| 4 | 40 | 1.60 | -40% | 1.07x | 1.34x |  0.983  |   0.855  |
| 4 | 50 | 1.38 | -49% | 1.15x | 1.39x |  0.98   |   0.845  |
| 4 | 75 | 0.85 | -70% | 1.39x | 1.84x |  0.974  |   0.808  |
| 8 | 0 (baseline) | 2.59 | -    | 1.00x | 1.00x | 0.986 | 0.853 |
| 8 | 5  | 2.43 | -8%  | 0.95x | 1.04x |  0.988  |   0.867  |
| 8 | 10 | 2.30 | -13% | 1.02x | 1.09x |  0.988  |   0.871  |
| 8 | 20 | 2.06 | -22% | 1.04x | 1.14x |  0.987  |   0.867  |
| 8 | 30 | 1.84 | -31% | 1.01x | 1.24x |  0.986  |   0.858  |
| 8 | 40 | 1.60 | -40% | 1.07x | 1.32x |  0.984  |   0.86  |
| 8 | 50 | 1.38 | -49% | 1.14x | 1.46x |  0.982  |   0.851  |
| 8 | 75 | 0.85 | -70% | 1.48x | 1.88x |  0.971  |   0.812  |
| 16 | 0 (baseline) | 2.59 | -    | 1.00x | 1.00x | 0.983 | 0.854 |
| 16 | 5  | 2.43 | -8%  | 0.97x | 1.08x |  0.987  |  0.866  |
| 16 | 10 | 2.30 | -13% | 1.01x | 1.09x |  0.987  |  0.87   |
| 16 | 20 | 2.06 | -22% | 1.05x | 1.14x |  0.985  |  0.861  |
| 16 | 30 | 1.84 | -31% | 0.95x | 1.26x |  0.983  |  0.857  |
| 16 | 40 | 1.60 | -40% | 1.07x | 1.32x |  0.983  |  0.854  |
| 16 | 50 | 1.38 | -49% | 1.13x | 1.45x |  0.978  |  0.844  |
| 16 | 75 | 0.85 | -70% | 1.21x | 1.78x |  0.972  |  0.821  |


