import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import io
import contextlib

import torch

# Reuse functions from export_compare_onnx to avoid duplicate logic
try:
    import export_compare_onnx as eco
except ImportError:
    print("ERROR: Could not import export_compare_onnx.py. Ensure this script resides in the same directory.")
    sys.exit(1)


def ensure_dependencies():
    missing = []
    try:
        import onnx  # noqa: F401
    except Exception:
        missing.append('onnx')
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        missing.append('onnxruntime')
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Install them in the active environment, e.g.:")
        print("  pip install onnx onnxruntime")
        sys.exit(2)


def discover_models(runs_dir: Path) -> Dict[int, Dict[str, Path]]:
    """
    Discover finetuned and pruned model directories.
    Returns mapping: {batch_size: { 'finetune': Path(best.pt), 'pruned': {prune_pct: Path(best.pt), ...}}}
    """
    pattern_finetune = re.compile(r"finetune_b(\d+)$")
    pattern_pruned = re.compile(r"pruned_b(\d+)_([0-9]+\.[0-9]+)%_after_finetune$")
    result: Dict[int, Dict[str, Dict]] = {}
    for child in runs_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        m_f = pattern_finetune.match(name)
        if m_f:
            b = int(m_f.group(1))
            best = child / 'weights' / 'best.pt'
            if best.exists():
                entry = result.setdefault(b, {'finetune': None, 'pruned': {}})
                entry['finetune'] = best
            continue
        m_p = pattern_pruned.match(name)
        if m_p:
            b = int(m_p.group(1))
            pct = float(m_p.group(2))
            best = child / 'weights' / 'best.pt'
            if best.exists():
                entry = result.setdefault(b, {'finetune': None, 'pruned': {}})
                entry['pruned'][pct] = best
    return result


def benchmark_pair(orig_pt: Path, pruned_pt: Path, imgsz: int, batch: int, iters: int, device: str, dynamic: bool, simplify: bool,
                   cache: Dict[Tuple[str, int, bool, bool], Path]) -> Dict:
    """Benchmark a single original/pruned pair, caching ONNX exports to avoid duplication."""
    # Export or reuse ONNX paths
    key_orig = (str(orig_pt), imgsz, dynamic, simplify)
    key_pruned = (str(pruned_pt), imgsz, dynamic, simplify)
    if key_orig not in cache:
        cache[key_orig] = eco.export_to_onnx(str(orig_pt), imgsz, dynamic, simplify)
    if key_pruned not in cache:
        cache[key_pruned] = eco.export_to_onnx(str(pruned_pt), imgsz, dynamic, simplify)

    orig_onnx = cache[key_orig]
    pruned_onnx = cache[key_pruned]

    # PyTorch latency (average ms for full batch)
    orig_torch_ms = eco.benchmark_pytorch(str(orig_pt), imgsz, batch, iters, device)
    pruned_torch_ms = eco.benchmark_pytorch(str(pruned_pt), imgsz, batch, iters, device)

    # ONNX latency (ms per image)
    try:
        orig_onnx_ms, orig_provider = eco.benchmark_onnx(orig_onnx, imgsz, batch, iters)
    except Exception:
        orig_onnx = eco.export_to_onnx(str(orig_pt), imgsz, False, False)
        orig_onnx_ms, orig_provider = eco.benchmark_onnx(orig_onnx, imgsz, batch, iters)
    try:
        pruned_onnx_ms, pruned_provider = eco.benchmark_onnx(pruned_onnx, imgsz, batch, iters)
    except Exception:
        pruned_onnx = eco.export_to_onnx(str(pruned_pt), imgsz, False, False)
        pruned_onnx_ms, pruned_provider = eco.benchmark_onnx(pruned_onnx, imgsz, batch, iters)

    return {
        'orig_pt': str(orig_pt.resolve()),
        'pruned_pt': str(pruned_pt.resolve()),
        'orig_onnx': str(orig_onnx),
        'pruned_onnx': str(pruned_onnx),
        'pytorch_original_avg_ms': orig_torch_ms,
        'pytorch_pruned_avg_ms': pruned_torch_ms,
        'onnx_original_avg_ms_per_image': orig_onnx_ms,
        'onnx_pruned_avg_ms_per_image': pruned_onnx_ms,
        'onnx_original_provider': orig_provider,
        'onnx_pruned_provider': pruned_provider,
        'speedup_pytorch_pruned_vs_orig': orig_torch_ms / pruned_torch_ms if pruned_torch_ms else None,
        'speedup_onnx_pruned_vs_orig': orig_onnx_ms / pruned_onnx_ms if pruned_onnx_ms else None,
        'speedup_onnx_orig_vs_pytorch_orig': orig_torch_ms / orig_onnx_ms if orig_onnx_ms else None,
        'speedup_onnx_pruned_vs_pytorch_pruned': pruned_torch_ms / pruned_onnx_ms if pruned_onnx_ms else None,
    }


def extract_model_stats(model_path: Path, imgsz: int) -> Tuple[float, float]:
    """Return (params_millions, gflops) for a YOLO model.

    GFLOPs estimated via a single forward pass counting Conv2d and Linear operations.
    """
    from ultralytics import YOLO
    ymodel = YOLO(str(model_path)).model
    params_m = sum(p.numel() for p in ymodel.parameters()) / 1e6

    flops = 0
    hooks = []

    def hook_conv(module, inp, out):
        # inp[0]: (B, C_in, H_in, W_in); out: (B, C_out, H_out, W_out)
        if isinstance(out, torch.Tensor):
            b, c_out, h_out, w_out = out.shape
            c_in = module.in_channels
            k_h, k_w = module.kernel_size
            groups = module.groups
            # Multiply-Add counts as 2 operations; here we count MACs only (common in GFLOPs reporting)
            conv_flops = k_h * k_w * (c_in / groups) * c_out * h_out * w_out
            nonlocal flops
            flops += conv_flops

    def hook_linear(module, inp, out):
        if isinstance(out, torch.Tensor):
            in_features = module.in_features
            out_features = module.out_features
            nonlocal flops
            flops += in_features * out_features

    for m in ymodel.modules():
        if hasattr(m, 'weight'):
            if m.__class__.__name__ == 'Conv2d':
                hooks.append(m.register_forward_hook(hook_conv))
            elif m.__class__.__name__ == 'Linear':
                hooks.append(m.register_forward_hook(hook_linear))

    with torch.no_grad():
        dummy = torch.randn(1, 3, imgsz, imgsz, device=next(ymodel.parameters()).device)
        _ = ymodel(dummy)

    for h in hooks:
        h.remove()

    gflops = flops / 1e9
    return params_m, gflops


def write_csv(csv_path: Path, rows: List[Dict]):
    if not rows:
        return
    fieldnames = ['batch_size', 'prune_pct'] + [k for k in rows[0].keys() if k not in ('batch_size', 'prune_pct')]
    write_header = not csv_path.exists()
    with csv_path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser(description="Batch compare original vs pruned YOLO models across prune levels.")
    parser.add_argument('--runs_dir', type=Path, default=Path('runs'), help='Root directory containing finetune/pruned subdirectories')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--simplify', action='store_true')
    parser.add_argument('--output_dir', type=Path, default=Path('logs') / 'compare_all')
    parser.add_argument('--limit_batches', type=int, nargs='*', help='Restrict to these batch sizes (e.g. 16 8)')
    parser.add_argument('--limit_prunes', type=float, nargs='*', help='Restrict to these prune percentages (e.g. 5 10 20 30 40 50 75)')
    parser.add_argument('--stop_after', type=int, default=None, help='Stop after N comparisons (debug)')
    parser.add_argument('--json', action='store_true', help='Write aggregated JSON summary')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing aggregated files')
    parser.add_argument('--summary_md', action='store_true', help='Write Markdown summary tables per batch')
    args = parser.parse_args()

    ensure_dependencies()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    csv_path = args.output_dir / f'benchmarks_img{args.imgsz}_b{args.batch}_{timestamp}.csv'
    json_path = args.output_dir / f'benchmarks_img{args.imgsz}_b{args.batch}_{timestamp}.json'
    log_path = args.output_dir / f'run_{timestamp}.log'

    with log_path.open('w') as log:
        def log_print(*msg):
            line = ' '.join(str(m) for m in msg)
            print(line)
            log.write(line + '\n')

        log_print(f"Device: {args.device}")
        log_print(f"Config: imgsz={args.imgsz} batch={args.batch} iters={args.iters} dynamic={args.dynamic} simplify={args.simplify}")
        discovered = discover_models(args.runs_dir)
        if not discovered:
            log_print("No models discovered.")
            return
        log_print(f"Discovered batch sizes: {sorted(discovered.keys())}")

        comparisons_done = 0
        cache: Dict[Tuple[str, int, bool, bool], Path] = {}
        aggregated_rows: List[Dict] = []
        json_rows: List[Dict] = []

        summary_tables: List[str] = []
        for batch_size, info in sorted(discovered.items()):
            if args.limit_batches and batch_size not in args.limit_batches:
                continue
            orig_pt = info.get('finetune')
            if not orig_pt:
                log_print(f"Skipping batch {batch_size}: no finetune best.pt")
                continue
            pruned_map = info.get('pruned', {})
            if not pruned_map:
                log_print(f"Skipping batch {batch_size}: no pruned models")
                continue
            # Extract baseline stats once
            params_m_orig, gflops_orig = extract_model_stats(orig_pt, args.imgsz)
            log_print(f"Baseline stats batch={batch_size}: params={params_m_orig:.3f}M, GFLOPs={gflops_orig:.2f}")
            log_print(f"Processing batch size {batch_size} with {len(pruned_map)} pruned variants")
            batch_rows_for_table = []
            for prune_pct, pruned_pt in sorted(pruned_map.items()):
                if args.limit_prunes and prune_pct not in args.limit_prunes:
                    continue
                log_print(f"Benchmarking batch={batch_size} prune={prune_pct}% ...")
                start = time.time()
                metrics = benchmark_pair(orig_pt, pruned_pt, args.imgsz, args.batch, args.iters, args.device, args.dynamic, args.simplify, cache)
                duration = time.time() - start
                params_m_pruned, gflops_pruned = extract_model_stats(pruned_pt, args.imgsz)
                flops_reduction_pct = (gflops_pruned - gflops_orig) / gflops_orig * 100 if (gflops_pruned == gflops_pruned and gflops_orig) else float('nan')
                row = {
                    'batch_size': batch_size,
                    'prune_pct': prune_pct,
                    **metrics
                }
                # Add stats
                row.update({
                    'params_m_orig': params_m_orig,
                    'params_m_pruned': params_m_pruned,
                    'gflops_orig': gflops_orig,
                    'gflops_pruned': gflops_pruned,
                    'flops_reduction_pct': flops_reduction_pct,
                })
                aggregated_rows.append(row)
                json_rows.append(row)
                if row['speedup_pytorch_pruned_vs_orig']:
                    log_print(
                        f"Done prune={prune_pct}% in {duration:.1f}s | params={params_m_pruned:.3f}M | GFLOPs={gflops_pruned:.2f} | FLOPs_Reduction={flops_reduction_pct:.0f}% | PT_speedup={row['speedup_pytorch_pruned_vs_orig']:.2f}x | ONNX_speedup={row['speedup_onnx_pruned_vs_orig']:.2f}x"
                    )
                else:
                    log_print(f"Done prune={prune_pct}% in {duration:.1f}s | params={params_m_pruned:.3f}M | GFLOPs={gflops_pruned:.2f}")
                comparisons_done += 1
                batch_rows_for_table.append({
                    'prune_pct': prune_pct,
                    'params_m_pruned': params_m_pruned,
                    'flops_reduction_pct': flops_reduction_pct,
                    'pytorch_speedup': row['speedup_pytorch_pruned_vs_orig'],
                    'onnx_speedup': row['speedup_onnx_pruned_vs_orig'],
                })
                if args.stop_after and comparisons_done >= args.stop_after:
                    log_print("Reached stop_after limit; terminating early.")
                    break
            if args.stop_after and comparisons_done >= args.stop_after:
                break
            # Build markdown table for this batch
            if args.summary_md and batch_rows_for_table:
                md = [f"### Batch size {batch_size}",
                      "| Pruning % | Params (M) | FLOPs Reduction | PyTorch Speedup | ONNX Speedup |",
                      "|-----------|-----------:|----------------:|----------------:|-------------:|"]
                # Baseline row
                md.append(f"| 0% (baseline) | {params_m_orig:.2f} | - | 1.00x | 1.00x |")
                for r in sorted(batch_rows_for_table, key=lambda x: x['prune_pct']):
                    flops_red = f"{r['flops_reduction_pct']:.0f}%" if r['flops_reduction_pct'] == r['flops_reduction_pct'] else "-"
                    pt_speed = f"{r['pytorch_speedup']:.2f}x" if r['pytorch_speedup'] else "-"
                    onnx_speed = f"{r['onnx_speedup']:.2f}x" if r['onnx_speedup'] else "-"
                    md.append(f"| {r['prune_pct']:.0f}% | {r['params_m_pruned']:.2f} | {flops_red} | {pt_speed} | {onnx_speed} |")
                summary_tables.append('\n'.join(md))

        if aggregated_rows:
            write_csv(csv_path, aggregated_rows)  # Always append for fresh timestamped file
            log_print(f"CSV written: {csv_path}")
        if args.json and json_rows:
            if json_path.exists() and not args.overwrite:
                log_print(f"JSON exists and overwrite disabled: {json_path}")
            else:
                with json_path.open('w') as jf:
                    json.dump(json_rows, jf, indent=2)
                log_print(f"JSON written: {json_path}")
        if args.summary_md and summary_tables:
            md_path = args.output_dir / f'summary_{timestamp}.md'
            with md_path.open('w') as mf:
                mf.write('\n\n'.join(summary_tables) + '\n')
            log_print(f"Markdown summary written: {md_path}")
        log_print(f"Total comparisons executed: {comparisons_done}")
        log_print("Completed batch comparisons.")


if __name__ == '__main__':
    main()
