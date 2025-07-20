import argparse
import math

from copy import deepcopy

import torch

from ultralytics import YOLO, __version__
from ultralytics.nn.modules import Detect, A2C2f

from ultralytics.utils import YAML
from ultralytics.utils.checks import check_yaml
from ultralytics.utils.torch_utils import initialize_weights, de_parallel

from c3k2_v2 import replace_c3k2_with_c3k2_v2
from train_and_save import train_v2
from baseline import train_model

from metrics import get_inference_time, count_parameters

import torch_pruning as tp

def prune(args):
    # Load the model yolov11n
    model = YOLO(args.model)

    # Replace the default training method with a custom one that supports pruning
    model.__setattr__("train_v2", train_v2.__get__(model))

    # Loads training configuration from YAML file
    pruning_cfg = YAML.load(check_yaml(args.cfg))
    batch_size = pruning_cfg['batch']

    # use coco128 dataset for 10 epochs fine-tuning each pruning iteration step
    # this part is only for sample code, number of epochs should be included in config file
    pruning_cfg['data'] = "coco128.yaml"
    pruning_cfg['epochs'] = 1
    pruning_cfg['lr0'] = 0.001  # Lower initial learning rate
    pruning_cfg['warmup_epochs'] = 3  # Add warmup

    model.model.train()  # Set to training mode
    replace_c3k2_with_c3k2_v2(model.model)  # Critical step!
    initialize_weights(model.model)  # Reset batch norm statistics

    for name, param in model.model.named_parameters():
        param.requires_grad = True

    example_inputs = torch.randn(1, 3, pruning_cfg["imgsz"], pruning_cfg["imgsz"]).to(model.device)
    macs_list, nparams_list, map_list, pruned_map_list = [], [], [], []
    base_macs, base_nparams = tp.utils.count_ops_and_params(model.model, example_inputs)

    # do validation before pruning model
    pruning_cfg['name'] = f"baseline_val"
    pruning_cfg['batch'] = 1
    validation_model = deepcopy(model)
    metric = validation_model.val(**pruning_cfg)
    init_map = metric.box.map
    macs_list.append(base_macs)
    nparams_list.append(100)
    map_list.append(init_map)
    pruned_map_list.append(init_map)
    print(pruned_map_list)
    print(f"Before Pruning: MACs={base_macs / 1e9: .5f} G, #Params={base_nparams / 1e6: .5f} M, mAP={init_map: .5f}")

    

    pruning_ratio = 1 - math.pow((1 - args.target_prune_rate), 1 / args.iterative_steps)
    
    
    for i in range(args.iterative_steps):

        model.model.train()
        for name, param in model.model.named_parameters():
            param.requires_grad = True

        ignored_layers = []
        unwrapped_parameters = []
        for m in model.model.modules():
            if isinstance(m, (Detect,)):
                ignored_layers.append(m)
            elif isinstance(m, A2C2f):
                ignored_layers.append(m)    
  

        print("[DEBUG] Initializing GroupNormPruner...")

        # from ultralytics.nn.tasks import DetectionModel

        # def bypass_forward(self, x, *args, **kwargs):
        #     """Bypass problematic forward logic for FX tracing"""
        #     if isinstance(x, dict):
        #         return self.loss(x, *args, **kwargs)
        #     # Use the correct method signature
        #     return self.predict(x, profile=False, visualize=False, embed=None)

        # DetectionModel.forward = bypass_forward
        # print("[DEBUG] Model structure:")
        # for name, module in model.model.named_modules():
        #     print(f"{name}: {type(module)}")

        pruner = tp.pruner.GroupNormPruner(
            model.model,
            example_inputs,
            importance=tp.importance.GroupMagnitudeImportance(),
            iterative_steps=3,
            pruning_ratio=pruning_ratio,
            ignored_layers=ignored_layers,
            unwrapped_parameters=unwrapped_parameters,
            # forward_fn= bypass_forward
        )

        # pruner = tp.pruner.MagnitudePruner(
        #     model.model,
        #     example_inputs,
        #     importance=tp.importance.MagnitudeImportance(),
        #     iterative_steps=1,
        #     pruning_ratio=pruning_ratio,
        #     ignored_layers=ignored_layers,
        #     unwrapped_parameters=unwrapped_parameters,
        # )
            
        print("[DEBUG] GroupNormPruner initialized successfully")    

        

        # Test regularization
        #output = model.model(example_inputs)
        #(output[0].sum() + sum([o.sum() for o in output[1]])).backward()
        #pruner.regularize(model.model)
        print("[DEBUG] Before pruner.step()")
        pruner.step()
        print("[DEBUG] After pruner.step()")

        # COMPREHENSIVE DEVICE SYNCHRONIZATION
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"[DEBUG] Target device: {device}")

        # Move model and all components
        model.model = model.model.to(device)

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
                # Standard PyTorch module with .to() method
                move_module_to_device(criterion, device)
                print("[DEBUG] Moved criterion to device using .to()")
            else:
                # Custom loss function - move internal tensors manually
                print(f"[DEBUG] Criterion type: {type(criterion)}")
                for attr_name in dir(criterion):
                    if not attr_name.startswith('_') and not callable(getattr(criterion, attr_name)):
                        attr_value = getattr(criterion, attr_name)
                        if isinstance(attr_value, torch.Tensor):
                            setattr(criterion, attr_name, attr_value.to(device))
                            print(f"[DEBUG] Moved {attr_name} tensor to device")
                print("[DEBUG] Moved criterion tensors manually")
        else:
            print("[DEBUG] model.model.criterion is None or doesn't exist, skipping")

        # pre fine-tuning validation
        pruning_cfg['name'] = f"step_{i}_pre_val"
        pruning_cfg['batch'] = 1
        validation_model.model = deepcopy(model.model)
        metric = validation_model.val(**pruning_cfg)
        pruned_map = metric.box.map
        pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(pruner.model, example_inputs.to(model.device))
        current_speed_up = float(macs_list[0]) / pruned_macs
        print(f"After pruning iter {i + 1}: MACs={pruned_macs / 1e9} G, #Params={pruned_nparams / 1e6} M, "
              f"mAP={pruned_map}, speed up={current_speed_up}")

        # fine-tuning
        for name, param in model.model.named_parameters():
            param.requires_grad = True
        pruning_cfg['name'] = f"step_{i}_finetune"
        pruning_cfg['batch'] = batch_size  # restore batch size
        model.train_v2(pruning=True, **pruning_cfg)

        # post fine-tuning validation
        pruning_cfg['name'] = f"step_{i}_post_val"
        pruning_cfg['batch'] = 1
        validation_model = YOLO(model.trainer.best)
        metric = validation_model.val(**pruning_cfg)
        current_map = metric.box.map
        print(f"After fine tuning mAP={current_map}")

        macs_list.append(pruned_macs)
        nparams_list.append(pruned_nparams / base_nparams * 100)
        pruned_map_list.append(pruned_map)
        map_list.append(current_map)

        # remove pruner after single iteration
        del pruner

        # if init_map - current_map > args.max_map_drop:
        #     print("Pruning early stop")
        #     break

    model.export(format='onnx')            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolo12n.pt', help='Pretrained pruning target model file')
    parser.add_argument('--cfg', default='default.yaml',
                        help='Pruning config file.'
                             ' This file should have same format with ultralytics/yolo/cfg/default.yaml')
    parser.add_argument('--iterative_steps', default=3, type=int, help='Total pruning iteration step')
    parser.add_argument('--target-prune-rate', default=0.3, type=float, help='Target pruning rate')
    parser.add_argument('--max-map-drop', default=0.2, type=float, help='Allowed maximum map drop after fine-tuning')

    args = parser.parse_args()

    prune(args)