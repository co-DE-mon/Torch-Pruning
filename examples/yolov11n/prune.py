import argparse
import torch
from pathlib import Path
import torch_pruning as tp

from ultralytics import YOLO, __version__
from ultralytics.nn.modules import C2PSA,Detect

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
    pruning_cfg['batch'] = 4
    pruning_cfg['model'] = args.model

    model.model.train()  # Set to training mode
    replace_c3k2_with_c3k2_v2(model.model)  # Critical step!
    initialize_weights(model.model)  # Reset batch norm statistics

    model.model.to(model.device)


    example_inputs = torch.randn(1, 3, pruning_cfg["imgsz"], pruning_cfg["imgsz"]).to(model.device)

    # flops, nparams = get_flops_and_params(model.model, example_inputs)
    # print(f"🔍 BEFORE PRUNING:")
    # print(f"   Total Parameters: {nparams / 1e6: .5f} M")
    
    # """  
    flops, nparams = get_flops_and_params(model.model, example_inputs)
    # macs= flops/2.0
    # mean_lat, std_lat = get_latency(model.model, example_inputs)
    # size = get_model_size(model.model)
    # fps= get_fps(model.model, example_inputs)
    # mem_bw = (size / 1024)* fps
    # inference_time= get_inference_time(model.model, example_inputs)

    print(f"🔍 BEFORE PRUNING:")
    print(f"   Total Parameters: {nparams / 1e6: .5f} M")
    # print(f"   MACs: {macs / 1e9: .5f} G")
    # print(f"   FLOPs: {flops / 1e9: .5f} G")
    # print(f"   Model Size: {size} MB")
    # print(f"   Memory Bandwidth: {mem_bw} GB/s ")
    # print(f"   FPS: {fps}  ")
    # print(f"   Inference: {inference_time} ")
    # print(f"   Latency: {mean_lat} ")
    # """
    pruning_ratio = 0.1127

    for i in range(args.iterative_steps):
        
        model.model.train()
        for name, param in model.model.named_parameters():
            param.requires_grad = True

        ignored_layers = []
        unwrapped_parameters = []
        for m in model.model.modules():
            # if isinstance(m, (Detect,)):
            #     ignored_layers.append(m)
            if isinstance(m, C2PSA):
                ignored_layers.append(m) 

        pruner = tp.pruner.GroupNormPruner(
            model.model,
            example_inputs,
            importance=tp.importance.GroupMagnitudeImportance(),
            iterative_steps=1,
            pruning_ratio=pruning_ratio,
            ignored_layers=ignored_layers,
            unwrapped_parameters=unwrapped_parameters
            )        
        
        pruner.step()

        # COMPREHENSIVE DEVICE SYNCHRONIZATION
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"[DEBUG] Target device: {device}")

        model.model = model.model.to(device)
        base_dir = Path(__file__).resolve().parent
        save_dir_before = base_dir / "runs" / "pruned_models"
        save_dir_after = base_dir / "runs"
         # Ensure paths exist
        save_dir_before.mkdir(parents=True, exist_ok=True)
        save_dir_before.mkdir(parents=True, exist_ok=True)
        pruned_model_path = save_dir_before / f"pruned_step_{i}_before_finetune.pt"
        torch.save(model.model.state_dict(), pruned_model_path)

        
        # fine-tuning
        for name, param in model.model.named_parameters():
            param.requires_grad = True
        pruning_cfg['name'] = f"pruned_step_{i}_after_finetune"
        pruning_cfg['epochs'] =args.postprune_epochs  # restore epoch size
        pruning_cfg['batch'] =4  # restore batch size
        pruning_cfg['project']=save_dir_after
        pruning_cfg['exist_ok']=True

        print("----------------- POST PRUNING FINETUNING ----------------- ")
        model.train_v2(pruning=True, **pruning_cfg)

        # flops, nparams = get_flops_and_params(model.model, example_inputs)
        # print(f"🔍 AFTER PRUNING:")
        # print(f"   Total Parameters: {nparams / 1e6: .5f} M")
        
        # """
        flops, nparams = get_flops_and_params(model.model, example_inputs)
        # macs= flops/2.0
        # mean_lat, std_lat = get_latency(model.model, example_inputs)
        # size = get_model_size(model.model)
        # fps= get_fps(model.model, example_inputs)
        # mem_bw = (size / 1024)* fps
        # inference_time= get_inference_time(model.model, example_inputs)

        print(f"🔍 AFTER PRUNING:")
        print(f"   Total Parameters: {nparams / 1e6: .5f} M")
        # print(f"   MACs: {macs / 1e9: .5f} G")
        # print(f"   FLOPs: {flops / 1e9: .5f} G")
        # print(f"   Model Size: {size} MB")
        # print(f"   Memory Bandwidth: {mem_bw} GB/s ")
        # print(f"   FPS: {fps}  ")
        # print(f"   Inference: {inference_time} ")
        # print(f"   Latency: {mean_lat} ")
        # """

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="examples/yolov11n/runs/fine_tuned_yolo11n/weights/best.pt", help='Model path')
    parser.add_argument('--data', default="examples/yolov11n/BCCD/data.yaml", help='DATA path')
    parser.add_argument('--cfg', default='default.yaml',
                        help='Pruning config file.'
                             ' This file should have same format with ultralytics/yolo/cfg/default.yaml')    
    parser.add_argument('--iterative_steps', default=1, type=int, help='Number of iterative steps') 
    parser.add_argument('--postprune_epochs', default=50, type=int, help='Number of epochs for post pruning finetuning') 

    args = parser.parse_args()

    pruned_model = prune(args)    