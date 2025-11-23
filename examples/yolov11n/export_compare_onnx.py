import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

try:
    from ultralytics import YOLO
except ImportError as e:
    raise SystemExit("Ultralytics not installed. Install with pip install ultralytics") from e


def export_to_onnx(model_path: str, imgsz: int, dynamic: bool, simplify: bool):
    model = YOLO(model_path)
    exported = model.export(format="onnx", imgsz=imgsz, dynamic=dynamic, simplify=simplify)
    # ultralytics returns a dict-like; attempt to find path
    onnx_path = None
    if isinstance(exported, (list, tuple)):
        for item in exported:
            if isinstance(item, (str, Path)) and str(item).endswith('.onnx'):
                onnx_path = Path(item)
                break
    elif isinstance(exported, dict):
        onnx_path = Path(exported.get('model', exported.get('path', '')))
    elif isinstance(exported, (str, Path)):
        onnx_path = Path(exported)
    # Fallback search
    if onnx_path is None or not onnx_path.exists():
        parent = Path(model_path).parent
        candidates = list(parent.glob('*.onnx'))
        if candidates:
            onnx_path = candidates[0]
    if onnx_path is None or not onnx_path.exists():
        raise RuntimeError(f"ONNX export failed for {model_path}")
    return onnx_path


def benchmark_pytorch(model_path: str, imgsz: int, batch: int, iters: int, device: str):
    model = YOLO(model_path)
    torch_model = model.model.to(device).eval()
    # Random input
    inp = torch.randn(batch, 3, imgsz, imgsz, device=device)
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = torch_model(inp)
    torch.cuda.synchronize() if device.startswith('cuda') else None
    start = time.time()
    for _ in range(iters):
        with torch.no_grad():
            _ = torch_model(inp)
    torch.cuda.synchronize() if device.startswith('cuda') else None
    elapsed = time.time() - start
    avg_ms = (elapsed / iters) * 1000
    return avg_ms


def benchmark_onnx(onnx_path: Path, imgsz: int, batch: int, iters: int):
    """Benchmark an ONNX model, gracefully handling fixed batch exports (e.g., batch=1) when a larger batch is requested.

    If the exported ONNX has a fixed first dimension (e.g. 1) and the caller requests batch>1, we benchmark using the fixed
    dimension and report per-image latency accordingly, logging a note instead of failing.
    """
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise SystemExit("onnxruntime not installed. Install with pip install onnxruntime-gpu") from e
    providers = []
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.append('CUDAExecutionProvider')
    providers.append('CPUExecutionProvider')

    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape
    # Determine effective batch: If model fixed batch (int) and differs from requested, use model batch.
    model_batch = input_shape[0] if isinstance(input_shape[0], int) else batch
    effective_batch = model_batch
    if isinstance(input_shape[0], int) and input_shape[0] != batch:
        print(f"[INFO] ONNX model has fixed batch={input_shape[0]} (requested {batch}); using fixed batch for benchmarking.")
    h = imgsz
    w = imgsz
    inp = np.random.randn(effective_batch, 3, h, w).astype(np.float32)
    # Warmup
    for _ in range(min(10, max(2, iters//10))):
        _ = session.run(None, {input_name: inp})
    start = time.time()
    for _ in range(iters):
        _ = session.run(None, {input_name: inp})
    elapsed = time.time() - start
    ms_per_image = (elapsed / iters) / effective_batch * 1000.0
    used_provider = session.get_providers()[0]
    return ms_per_image, used_provider


def main():
    parser = argparse.ArgumentParser(description="Export original and pruned YOLO model to ONNX and benchmark")
    parser.add_argument('--orig', required=True, help='Path to original model .pt')
    parser.add_argument('--pruned', required=True, help='Path to pruned model .pt')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', default='onnx_benchmark_results.json')
    parser.add_argument('--dynamic', action='store_true', help='Enable dynamic axes in ONNX export')
    parser.add_argument('--simplify', action='store_true', help='Enable ONNX graph simplification')
    args = parser.parse_args()

    print(f"Using device: {args.device}")
    print("Exporting original model to ONNX...")
    orig_onnx = export_to_onnx(args.orig, args.imgsz, args.dynamic, args.simplify)
    print(f"Original ONNX: {orig_onnx}")
    print("Exporting pruned model to ONNX...")
    pruned_onnx = export_to_onnx(args.pruned, args.imgsz, args.dynamic, args.simplify)
    print(f"Pruned ONNX: {pruned_onnx}")

    print("Benchmarking PyTorch original...")
    orig_torch_ms = benchmark_pytorch(args.orig, args.imgsz, args.batch, args.iters, args.device)
    print("Benchmarking PyTorch pruned...")
    pruned_torch_ms = benchmark_pytorch(args.pruned, args.imgsz, args.batch, args.iters, args.device)

    print("Benchmarking ONNX original...")
    try:
        orig_onnx_ms, orig_provider = benchmark_onnx(orig_onnx, args.imgsz, args.batch, args.iters)
    except Exception as e:
        print(f"[WARN] Original ONNX benchmark error: {e}. Attempting re-export without dynamic/simplify.")
        orig_onnx = export_to_onnx(args.orig, args.imgsz, False, False)
        orig_onnx_ms, orig_provider = benchmark_onnx(orig_onnx, args.imgsz, args.batch, args.iters)
    print("Benchmarking ONNX pruned...")
    try:
        pruned_onnx_ms, pruned_provider = benchmark_onnx(pruned_onnx, args.imgsz, args.batch, args.iters)
    except Exception as e:
        print(f"[WARN] Pruned ONNX benchmark error: {e}. Attempting re-export without dynamic/simplify.")
        pruned_onnx = export_to_onnx(args.pruned, args.imgsz, False, False)
        pruned_onnx_ms, pruned_provider = benchmark_onnx(pruned_onnx, args.imgsz, args.batch, args.iters)

    results = {
        'config': {
            'image_size': args.imgsz,
            'batch': args.batch,
            'iterations': args.iters,
            'device': args.device,
        },
        'paths': {
            'orig_pt': os.path.abspath(args.orig),
            'pruned_pt': os.path.abspath(args.pruned),
            'orig_onnx': str(orig_onnx),
            'pruned_onnx': str(pruned_onnx),
        },
        'latency_ms': {
            'pytorch_original_avg_ms': orig_torch_ms,
            'pytorch_pruned_avg_ms': pruned_torch_ms,
            'onnx_original_avg_ms': orig_onnx_ms,
            'onnx_pruned_avg_ms': pruned_onnx_ms,
        },
        'providers': {
            'onnx_original_provider': orig_provider,
            'onnx_pruned_provider': pruned_provider,
        },
        'speedup': {
            'pytorch_pruned_vs_orig': orig_torch_ms / pruned_torch_ms if pruned_torch_ms else None,
            'onnx_pruned_vs_orig': orig_onnx_ms / pruned_onnx_ms if pruned_onnx_ms else None,
            'onnx_orig_vs_pytorch_orig': orig_torch_ms / orig_onnx_ms if orig_onnx_ms else None,
            'onnx_pruned_vs_pytorch_pruned': pruned_torch_ms / pruned_onnx_ms if pruned_onnx_ms else None,
        }
    }

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print("=== Summary ===")
    for k, v in results['latency_ms'].items():
        print(f"{k}: {v:.3f} ms")
    print("Providers:", results['providers'])
    print("Speedup ratios:")
    for k, v in results['speedup'].items():
        print(f"{k}: {v:.3f}" if v else f"{k}: None")
    print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
