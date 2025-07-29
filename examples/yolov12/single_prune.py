import argparse
import math

import torch

from ultralytics import YOLO, __version__
from ultralytics.nn.modules import Detect, A2C2f

from ultralytics.utils import YAML
from ultralytics.utils.checks import check_yaml
from ultralytics.utils.torch_utils import initialize_weights

from c3k2_v2 import replace_c3k2_with_c3k2_v2

from metrics import get_flops_and_params, get_latency, get_model_size, get_fps, get_inference_time

import torch_pruning as tp

def prune(args):
    # Load the model yolov11n
    model = YOLO(args.model)

    # Loads training configuration from YAML file
    pruning_cfg = YAML.load(check_yaml(args.cfg))
    pruning_cfg['data'] = "coco128.yaml"

    replace_c3k2_with_c3k2_v2(model.model)  # Critical step!
    initialize_weights(model.model)  # Reset batch norm statistics

    example_inputs = torch.randn(1, 3, pruning_cfg["imgsz"], pruning_cfg["imgsz"]).to(model.device)

    """  
    flops, nparams = get_flops_and_params(model.model, example_inputs)
    macs= flops/2.0
    mean_lat, std_lat = get_latency(model.model, example_inputs)
    size = get_model_size(model.model)
    fps= get_fps(model.model, example_inputs)
    mem_bw = (size / 1024)* fps
    inference_time= get_inference_time(model.model, example_inputs)

    print(f"🔍 BEFORE PRUNING:")
    print(f"   Total Parameters: {nparams / 1e6: .5f} M")
    print(f"   MACs: {macs / 1e9: .5f} G")
    print(f"   FLOPs: {flops / 1e9: .5f} G")
    print(f"   Model Size: {size} MB")
    print(f"   Memory Bandwidth: {mem_bw} GB/s ")
    print(f"   FPS: {fps}  ")
    print(f"   Inference: {inference_time} ")
    print(f"   Latency: {mean_lat} ")
    """
    pruning_ratio = 0.3
    ignored_layers = []

    unwrapped_parameters = []
    for m in model.model.modules():
        # if isinstance(m, (Detect,)):
        #     ignored_layers.append(m)
        if isinstance(m, A2C2f):
            ignored_layers.append(m) 

    pruner = tp.pruner.GroupNormPruner(
        model.model,
        example_inputs,
        importance=tp.importance.GroupMagnitudeImportance(),
        iterative_steps=3,
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored_layers,
        unwrapped_parameters=unwrapped_parameters
        )        
    
    pruner.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolo12n.pt', help='Model path')
    parser.add_argument('--cfg', default='default.yaml',
                        help='Pruning config file.'
                             ' This file should have same format with ultralytics/yolo/cfg/default.yaml')    
    args = parser.parse_args()

    pruned_model = prune(args)    