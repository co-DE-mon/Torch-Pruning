import time
import numpy as np

def get_map_metrics(model):
    """
    Validate the model and return mAP metrics
    """
    print("Validating model...")
    metrics = model.val()
    
    # Print metrics
    print("\n=== Training Results ===")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP75: {metrics.box.map75:.4f}")
    
    return metrics

def get_inference_time(model):
    """
    Test inference speed and return average inference time in milliseconds
    """
    print("\n=== Inference Speed Test ===")
    
    # Create a dummy image for speed test
    dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Warm up the model
    print("Warming up model...")
    for _ in range(10):
        model.predict(dummy_image, verbose=False)
    
    # Time inference over multiple runs
    print("Testing inference speed...")
    inference_times = []
    for _ in range(100):
        start = time.time()
        model.predict(dummy_image, verbose=False)
        inference_times.append((time.time() - start) * 1000)  # Convert to ms
    
    avg_inference_time = np.mean(inference_times)
    print(f"Average inference time: {avg_inference_time:.2f} ms")
    
    return avg_inference_time

def count_parameters(model):
    print( sum(p.numel() for p in model.parameters() if p.requires_grad))