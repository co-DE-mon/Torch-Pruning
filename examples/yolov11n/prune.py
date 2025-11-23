import argparse
import torch
from pathlib import Path
import torch_pruning as tp
import math

from ultralytics import YOLO, __version__
from ultralytics.nn.modules import C2PSA,Detect,Attention

from ultralytics.utils import YAML
from ultralytics.utils.checks import check_yaml
from ultralytics.utils.torch_utils import initialize_weights

from c3k2_v2 import replace_c3k2_with_c3k2_v2
from train_v2 import train_v2
from finetune import fine_tune_yolov11n

from metrics import get_flops_and_params, get_latency, get_model_size, get_fps, get_inference_time


def prune(args):
    # Load the model
    model = YOLO(args.model)

    # Replace the default training method with a custom one that supports pruning
    model.__setattr__("train_v2", train_v2.__get__(model))

    # Loads training configuration from YAML file
    pruning_cfg = YAML.load(check_yaml(args.cfg))
    pruning_cfg['data'] = args.data
    pruning_cfg['epochs'] = 10
    pruning_cfg['lr0'] = 0.001  # Lower initial learning rate
    pruning_cfg['batch'] = args.batch_size
    pruning_cfg['model'] = args.model

    model.model.train()  # Set to training mode
    replace_c3k2_with_c3k2_v2(model.model) 
    # initialize_weights(model.model)  

    for name, param in model.model.named_parameters():
        param.requires_grad = True

    # Establish a single device early and ensure model + example inputs match
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.model.to(device)
    example_inputs = torch.randn(1, 3, pruning_cfg["imgsz"], pruning_cfg["imgsz"], device=device)

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

    for m in model.model.modules():
        if isinstance(m, Attention):
            fixed_params = sum(p.numel() for p in m.parameters(recurse=True))

    target_prunable = (nparams*args.target_prune_rate)/(nparams-fixed_params)
    if target_prunable >= 1.0:
        target_prunable = 0.99
    pruning_ratio = 1 - math.pow((1 - target_prunable), 1 / args.iterative_steps)

    for i in range(args.iterative_steps):
        model.model.train()
        for name, param in model.model.named_parameters():
            param.requires_grad = True

        ignored_layers = []
        unwrapped_parameters = []
        for m in model.model.modules():
            if isinstance(m, Attention):
                ignored_layers.append(m)

        # Guarantee example_inputs resides on same device as model weights before tracing
        model_device = next(model.model.parameters()).device
        if example_inputs.device != model_device:
            example_inputs = example_inputs.to(model_device)

        pruner = tp.pruner.GroupNormPruner(
            model.model,
            example_inputs,
            importance=tp.importance.GroupMagnitudeImportance(),
            iterative_steps=args.iterative_steps,
            pruning_ratio=pruning_ratio,
            ignored_layers=ignored_layers,
            unwrapped_parameters=unwrapped_parameters
        )
        pruner.step()

        # COMPREHENSIVE DEVICE SYNCHRONIZATION
        # Reaffirm model on target device (should already be) and keep example_inputs consistent
        model.model = model.model.to(device)
        example_inputs = example_inputs.to(device)
        # Ensure all modules and their attributes are on the correct device
        def move_module_to_device(module, target_device):
            module.to(target_device)
            for attr_name in dir(module):
                if not attr_name.startswith('_') and not callable(getattr(module, attr_name)):
                    attr_value = getattr(module, attr_name)
                    if isinstance(attr_value, torch.Tensor):
                        setattr(module, attr_name, attr_value.to(target_device))

        for module in model.model.modules():
            move_module_to_device(module, device)

        # Handle loss function specifically - LOSS-AWARE VERSION
        if hasattr(model.model, 'criterion') and model.model.criterion is not None:
            criterion = model.model.criterion
            if hasattr(criterion, 'to'):
                move_module_to_device(criterion, device)
            else:
                print(f"[DEBUG] Criterion type: {type(criterion)}")
                for attr_name in dir(criterion):
                    if not attr_name.startswith('_') and not callable(getattr(criterion, attr_name)):
                        attr_value = getattr(criterion, attr_name)
                        if isinstance(attr_value, torch.Tensor):
                            setattr(criterion, attr_name, attr_value.to(device))
        else:
            print("model.model.criterion is None or doesn't exist, skipping")

        base_dir = Path(__file__).resolve().parent
        save_dir_before = base_dir / "runs" / "pruned_models"
        save_dir_after = base_dir / "runs"
        save_dir_before.mkdir(parents=True, exist_ok=True)
        pruned_model_path = save_dir_before / f"pruned_b{args.batch_size}_{int(args.target_prune_rate*100)}.0%_step_{i}_before_finetune.pt"
        torch.save(model.model.state_dict(), pruned_model_path)

        target_prunable = (nparams*args.target_prune_rate)/(nparams-fixed_params)
        if target_prunable >= 1.0:
            target_prunable = 0.99
        pruning_ratio = 1 - math.pow((1 - target_prunable), 1 / args.iterative_steps)

    # fine-tuning
    for name, param in model.model.named_parameters():
        param.requires_grad = True
    pruning_cfg['name'] = f"pruned_b{args.batch_size}_{int(args.target_prune_rate*100)}.0%_after_finetune"
    pruning_cfg['epochs'] = args.postprune_epochs
    pruning_cfg['batch'] = args.batch_size
    pruning_cfg['project'] = save_dir_after
    pruning_cfg['exist_ok'] = True

    print("----------------- POST PRUNING FINETUNING ----------------- ")
    model.train_v2(pruning=True, **pruning_cfg)

    # After training the YOLO wrapper may reload weights onto CPU; re-sync to pruning device
    model.model.to(device)
    example_inputs = example_inputs.to(device)

    flops, nparams = get_flops_and_params(model.model, example_inputs)
    macs= flops/2.0
    mean_lat, std_lat = get_latency(model.model, example_inputs)
    size = get_model_size(model.model)
    fps= get_fps(model.model, example_inputs)
    mem_bw = (size / 1024)* fps
    inference_time= get_inference_time(model.model, example_inputs)

    print(f"🔍 AFTER PRUNING:")
    print(f"   Total Parameters: {nparams / 1e6: .5f} M")

    print(f"   MACs: {macs / 1e9: .5f} G")
    print(f"   FLOPs: {flops / 1e9: .5f} G")
    print(f"   Model Size: {size} MB")
    print(f"   Memory Bandwidth: {mem_bw} GB/s ")
    print(f"   FPS: {fps}  ")
    print(f"   Inference: {inference_time} ")
    print(f"   Latency: {mean_lat} ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="examples/yolov11n/runs/fine_tuned_yolo11n/weights/best.pt", help='Model path')
    parser.add_argument('--data', default="examples/yolov11n/BCCD/data.yaml", help='DATA path')
    parser.add_argument('--cfg', default='default.yaml',
                        help='Pruning config file.'
                             ' This file should have same format with ultralytics/yolo/cfg/default.yaml')    
    parser.add_argument('--iterative_steps', default=2, type=int, help='Number of iterative steps') 
    parser.add_argument('--postprune_epochs', default=2, type=int, help='Number of epochs for post pruning finetuning') 
    parser.add_argument('--batch_size', default=4, type=int, help='Batch Size')
    parser.add_argument('--target_prune_rate', default=0.5, type=float, help='Target pruning rate')

    args = parser.parse_args()
    pruned_model = prune(args)