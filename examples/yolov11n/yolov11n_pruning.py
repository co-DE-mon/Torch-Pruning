import argparse
import math
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from ultralytics import YOLO, __version__
from ultralytics.nn.modules import Detect, C3k2, Conv, Bottleneck, C2PSA, C2f

from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.engine.model import Model
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import YAML, LOGGER, RANK, DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS
from ultralytics.utils.checks import check_yaml
from ultralytics.utils.torch_utils import initialize_weights, de_parallel

import torch_pruning as tp

def save_pruning_performance_graph(x, y1, y2, y3):
    """
    Draw performance change graph
    Parameters
    ----------
    x : List
        Parameter numbers of all pruning steps
    y1 : List
        mAPs after fine-tuning of all pruning steps
    y2 : List
        MACs of all pruning steps
    y3 : List
        mAPs after pruning (not fine-tuned) of all pruning steps

    Returns
    -------

    """
    try:
        plt.style.use("ggplot")
    except:
        pass

    x, y1, y2, y3 = np.array(x), np.array(y1), np.array(y2), np.array(y3)
    y2_ratio = y2 / y2[0]

    # create the figure and the axis object
    fig, ax = plt.subplots(figsize=(8, 6))

    # plot the pruned mAP and recovered mAP
    ax.set_xlabel('Pruning Ratio')
    ax.set_ylabel('mAP')
    ax.plot(x, y1, label='recovered mAP')
    ax.scatter(x, y1)
    ax.plot(x, y3, color='tab:gray', label='pruned mAP')
    ax.scatter(x, y3, color='tab:gray')

    # create a second axis that shares the same x-axis
    ax2 = ax.twinx()

    # plot the second set of data
    ax2.set_ylabel('MACs')
    ax2.plot(x, y2_ratio, color='tab:orange', label='MACs')
    ax2.scatter(x, y2_ratio, color='tab:orange')

    # add a legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')

    ax.set_xlim(105, -5)
    ax.set_ylim(0, max(y1) + 0.05)
    ax2.set_ylim(0.05, 1.05)

    # calculate the highest and lowest points for each set of data
    max_y1_idx = np.argmax(y1)
    min_y1_idx = np.argmin(y1)
    max_y2_idx = np.argmax(y2)
    min_y2_idx = np.argmin(y2)
    max_y1 = y1[max_y1_idx]
    min_y1 = y1[min_y1_idx]
    max_y2 = y2_ratio[max_y2_idx]
    min_y2 = y2_ratio[min_y2_idx]

    # add text for the highest and lowest values near the points
    ax.text(x[max_y1_idx], max_y1 - 0.05, f'max mAP = {max_y1:.2f}', fontsize=10)
    ax.text(x[min_y1_idx], min_y1 + 0.02, f'min mAP = {min_y1:.2f}', fontsize=10)
    ax2.text(x[max_y2_idx], max_y2 - 0.05, f'max MACs = {max_y2 * y2[0] / 1e9:.2f}G', fontsize=10)
    ax2.text(x[min_y2_idx], min_y2 + 0.02, f'min MACs = {min_y2 * y2[0] / 1e9:.2f}G', fontsize=10)

    plt.title('Comparison of mAP and MACs with Pruning Ratio')
    plt.savefig('pruning_perf_change.png')


def infer_shortcut(bottleneck):
    c1 = bottleneck.cv1.conv.in_channels
    c2 = bottleneck.cv2.conv.out_channels
    return c1 == c2 and hasattr(bottleneck, 'add') and bottleneck.add

# # ---------------------- C3k2_v2 (Simple) ----------------------
class c3k2_v2_simple(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        # y = list(self.cv1(x).chunk(2, 1))
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

# ---------------------- C3k_v2 (-------) ----------------------
class c3k_v2(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e) 
        # self.cv0 = Conv(c1, self.c, 1, 1) 
        self.cv1 = Conv(c1, self.c, 1, 1) 
        self.cv2 = Conv(c1, self.c, 1, 1) 
        self.cv3 = Conv((1+n) * self.c, c2, 1, 1)  # final projection
        self.m = nn.Sequential(*[Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(2*n)])
    
    def forward(self, x):
        y = [self.cv1(x)]
        y.append(self.m(self.cv2(x)))
        return self.cv3(torch.cat(y, 1))

# ---------------------- C3k2_v2(Complex) ----------------------
class c3k2_v2_complex(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv0 = Conv(c1, c2, 1, 1)  # learnable shortcut path
        self.cv1 = Conv(c1, c2, 1, 1)  # main feature path
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList([C3k_v2(self.c, self.c, n=2, shortcut=shortcut, g=g, e=e) for _ in range(n)])
    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)  # Static iteration over ModuleList
        return self.cv2(torch.cat(y, 1))


def transfer_weights_c3k2(c2f, c2f_v2):
    c2f_v2.cv2 = c2f.cv2
    c2f_v2.m = c2f.m

    state_dict = c2f.state_dict()
    state_dict_v2 = c2f_v2.state_dict()

    # Transfer cv1 weights from C2f to cv0 and cv1 in C2f_v2
    old_weight = state_dict['cv1.conv.weight']
    half_channels = old_weight.shape[0] // 2
    state_dict_v2['cv0.conv.weight'] = old_weight[:half_channels]
    state_dict_v2['cv1.conv.weight'] = old_weight[half_channels:]

    # Transfer cv1 batchnorm weights and buffers from C2f to cv0 and cv1 in C2f_v2
    for bn_key in ['weight', 'bias', 'running_mean', 'running_var']:
        old_bn = state_dict[f'cv1.bn.{bn_key}']
        state_dict_v2[f'cv0.bn.{bn_key}'] = old_bn[:half_channels]
        state_dict_v2[f'cv1.bn.{bn_key}'] = old_bn[half_channels:]

    # Transfer remaining weights and buffers
    for key in state_dict:
        if not key.startswith('cv1.'):
            state_dict_v2[key] = state_dict[key]

    # Transfer all non-method attributes
    for attr_name in dir(c2f):
        attr_value = getattr(c2f, attr_name)
        if not callable(attr_value) and '_' not in attr_name:
            setattr(c2f_v2, attr_name, attr_value)

    c2f_v2.load_state_dict(state_dict_v2)

def replace_c3k2_with_c3k2_v2(module):
    """
    Replace C3k2 blocks with appropriate C3k2_v2 variants based on internal structure
    """
    for name, child_module in module.named_children():
        if isinstance(child_module, C3k2):
            shortcut = infer_shortcut(child_module.m[0])
            # Determine if it's simple or complex based on internal structure
            has_c3k_subblock= any('C3k' in str(type(m).__name__) for m in module.m) if hasattr(module, 'm') else False

            if has_c3k_subblock:
                # Complex C3k2 block with C3k sub-blocks
                print(f"Replacing complex C3k2 block: {name}")
                c3k2_v2 = c3k2_v2_complex(child_module.cv1.conv.in_channels, child_module.cv2.conv.out_channels,
                            n=len(child_module.m), shortcut=shortcut,
                            g=child_module.m[0].cv2.conv.groups,
                            e=child_module.c / child_module.cv2.conv.out_channels)
                transfer_weights_c3k2(child_module, c3k2_v2)
                setattr(module, name, c3k2_v2)
            else:
                c3k2_v2 = c3k2_v2_simple(child_module.cv1.conv.in_channels, child_module.cv2.conv.out_channels,
                            n=len(child_module.m), shortcut=shortcut,
                            g=child_module.m[0].cv2.conv.groups,
                            e=child_module.c / child_module.cv2.conv.out_channels)
                transfer_weights_c3k2(child_module, c3k2_v2)
                setattr(module, name, c3k2_v2)
            
        else:
            # Recursively process child modules
            replace_c3k2_with_c3k2_v2(child_module)

def save_model_v2(self: BaseTrainer):
    """
    Disabled half precision saving. originated from ultralytics/yolo/engine/trainer.py
    """
    ckpt = {
        'epoch': self.epoch,
        'best_fitness': self.best_fitness,
        'model': deepcopy(de_parallel(self.model)),
        'ema': deepcopy(self.ema.ema),
        'updates': self.ema.updates,
        'optimizer': self.optimizer.state_dict(),
        'train_args': vars(self.args),  # save as dict
        'date': datetime.now().isoformat(),
        'version': __version__}

    # Save last, best and delete
    torch.save(ckpt, self.last)
    if self.best_fitness == self.fitness:
        torch.save(ckpt, self.best)
    if (self.epoch > 0) and (self.save_period > 0) and (self.epoch % self.save_period == 0):
        torch.save(ckpt, self.wdir / f'epoch{self.epoch}.pt')
    del ckpt

def final_eval_v2(self: BaseTrainer):
    """
    originated from ultralytics/yolo/engine/trainer.py
    """
    for f in self.last, self.best:
        if f.exists():
            strip_optimizer_v2(f)  # strip optimizers
            if f is self.best:
                LOGGER.info(f'\nValidating {f}...')
                self.metrics = self.validator(model=f)
                self.metrics.pop('fitness', None)
                self.run_callbacks('on_fit_epoch_end')

def strip_optimizer_v2(f: Union[str, Path] = 'best.pt', s: str = '') -> None:
    """
    Disabled half precision saving. originated from ultralytics/yolo/utils/torch_utils.py
    """
    x = torch.load(f,weights_only=False)
    args = {**DEFAULT_CFG_DICT, **x['train_args']}  # combine model args with default args, preferring model args
    if x.get('ema'):
        x['model'] = x['ema']  # replace model with ema
    for k in 'optimizer', 'ema', 'updates':  # keys
        x[k] = None
    for p in x['model'].parameters():
        p.requires_grad = False
    x['train_args'] = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # strip non-default keys
    # x['model'].args = x['train_args']
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # filesize
    LOGGER.info(f"Optimizer stripped from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB")

def train_v2(self: YOLO, pruning=False, **kwargs):
    """
    Disabled loading new model when pruning flag is set. originated from ultralytics/yolo/engine/model.py
    """

    self._check_is_pytorch_model()
    if self.session:  # Ultralytics HUB session
        if any(kwargs):
            LOGGER.warning('WARNING ⚠️ using HUB training arguments, ignoring local training arguments.')
        kwargs = self.session.train_args

    overrides = self.overrides.copy()
    overrides.update(kwargs)

    if kwargs.get('cfg'):
        LOGGER.info(f"cfg file passed. Overriding default params with {kwargs['cfg']}.")
        overrides = YAML.load(check_yaml(kwargs['cfg']))
    
    overrides['mode'] = 'train'
    if not overrides.get('data'):
        raise AttributeError("Dataset required but missing, i.e. pass 'data=coco128.yaml'")
    if overrides.get('resume'):
        overrides['resume'] = self.ckpt_path

    if pruning:
        overrides['model'] = 'yolo11n.pt'

    self.task = overrides.get('task') or self.task
    task_map = self.task_map
    self.trainer = task_map[self.task]["trainer"](overrides=overrides, _callbacks=self.callbacks)
    
    if not pruning:
        if not overrides.get('resume'):  # manually set model only if not resuming
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model

    else:
        # pruning mode
        self.trainer.pruning = True
        self.trainer.model = self.model

        # replace some functions to disable half precision saving
        self.trainer.save_model = save_model_v2.__get__(self.trainer)
        self.trainer.final_eval = final_eval_v2.__get__(self.trainer)

    self.trainer.hub_session = self.session  # attach optional HUB session
    self.trainer.train()
    # Update model and cfg after training
    if RANK in (-1, 0):
        self.model, _ = attempt_load_one_weight(str(self.trainer.best))
        self.overrides = self.model.args
        self.metrics = getattr(self.trainer.validator, 'metrics', None)

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
    pruning_cfg['epochs'] = 15
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
            elif isinstance(m, C2PSA):  # YOLOv11's spatial attention blocks
                # You may want to be more conservative with C2PSA blocks
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
            iterative_steps=5,
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

        save_pruning_performance_graph(nparams_list, map_list, macs_list, pruned_map_list)

        if init_map - current_map > args.max_map_drop:
            print("Pruning early stop")
            break

    model.export(format='onnx')            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolo11n.pt', help='Pretrained pruning target model file')
    parser.add_argument('--cfg', default='default.yaml',
                        help='Pruning config file.'
                             ' This file should have same format with ultralytics/yolo/cfg/default.yaml')
    parser.add_argument('--iterative-steps', default=10, type=int, help='Total pruning iteration step')
    parser.add_argument('--target-prune-rate', default=0.05, type=float, help='Target pruning rate')
    parser.add_argument('--max-map-drop', default=0.2, type=float, help='Allowed maximum map drop after fine-tuning')

    args = parser.parse_args()

    prune(args)


























