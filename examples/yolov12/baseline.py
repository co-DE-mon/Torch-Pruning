from ultralytics import YOLO
import time
import torch

def train_model(m,e):
    """
    Fine-tune YOLOv11n on COCO128 dataset for 15 epochs
    Returns the trained model and training results
    """
    # Initialize model
    model = YOLO(m)
    
    # Start training
    print("Starting YOLOv11n training on COCO128...")
    start_time = time.time()
    
    # Training parameters
    model.train(
        data="coco128.yaml",
        epochs=e,
        imgsz=640,
        batch=16,  # Adjust based on your GPU memory
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),  # Use GPU 0, or 'cpu' for CPU training
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return model
