import time
import numpy as np
import torch_pruning as tp
import os
import tempfile
import torch

def get_mAP_metrics(yolo_model):
    metrics = yolo_model.val()
    mAP50 = metrics.box.map50
    mAP50_95 = metrics.box.map
    
    print(f"mAP@0.5: {mAP50:.4f}")
    print(f"mAP@0.5:0.95: {mAP50_95:.4f}")
    
    return mAP50, mAP50_95

def get_flops_and_params(model,example_inputs):
    flops, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    return flops , nparams

def get_latency(model, example_inputs):
    mean_lat, std_lat = tp.utils.benchmark.measure_latency(model,example_inputs)
    return mean_lat, std_lat

def get_fps(model, example_inputs):
    fps= tp.utils.benchmark.measure_fps(model,example_inputs)
    return fps

def get_model_size(model):
    """
    Get the size of a PyTorch model in MB.
    This function temporarily saves the model's state_dict and returns its size.
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        torch.save(model.state_dict(), tmp.name)
        size_mb = os.path.getsize(tmp.name) / (1024 * 1024)
    os.remove(tmp.name)
    return round(size_mb, 2)

def get_inference_time(model, example_inputs):
    """
    Test inference speed and return average inference time in milliseconds
    """
    # Warm up the model
    for _ in range(10):
        model.predict(example_inputs)
    
    # Time inference over multiple runs
    inference_times = []
    for _ in range(100):
        start = time.time()
        model.predict(example_inputs)
        inference_times.append((time.time() - start) * 1000)  # Convert to ms
    
    avg_inference_time = np.mean(inference_times)  
    return avg_inference_time