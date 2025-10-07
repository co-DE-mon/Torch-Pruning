from ultralytics import YOLO
from pathlib import Path
import argparse


def fine_tune_yolov11n(
    device: str = "0",  # GPU id or “cpu”
):
    """
    Fine-tune a YOLOv11n model.

    Args:
        pretrained_weights: path to a .pt file (e.g. "yolo11n.pt" or checkpoint)
        data_yaml: path to dataset YAML (with train/val paths and classes)
        epochs: number of epochs for fine-tuning
        img_size: input image size (square)
        batch_size: training batch size
        device: GPU id or "cpu"
        save_dir: directory to save results and final weights
    """

    # Load model (this downloads or loads the YOLOv11n architecture + weights)
    model = YOLO(args.model)
    base_dir = Path(__file__).resolve().parent
    save_dir = base_dir / "runs"
    # Ensure paths exist
    save_dir.mkdir(parents=True, exist_ok=True)

    # Fine-tune / train
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch_size,
        name="fine_tuned_yolo11n",
        project=save_dir,
        exist_ok=True,
        # You can pass other hyperparameters here, e.g. lr, optimizer args, augment, etc.
    )

    # Verify weights file was created
    weights_path = save_dir / "fine_tuned_yolo11n" / "weights" / "best.pt"
    if weights_path.is_file():
        print(f"Successfully created weights file at: {weights_path}")
    else:
        print(f"Warning: Weights file not found at expected location: {weights_path}")
        
    return model


if __name__ == "__main__":
    # Example usage
    pretrained = "yolo11n.pt"  # change to your weights
    data = str(Path(__file__).parent / "BCCD" / "data.yaml")  # define train/val split, classes, etc.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="yolo11n.pt", help='Model path')
    parser.add_argument('--data', default="examples/yolov11n/BCCD/data.yaml", help='DATA path')
    parser.add_argument('--epochs', default=50, type=int, help='number of epochs')    
    parser.add_argument('--batch_size', default=4, type=int, help='batch_size') 
    args = parser.parse_args()
    # Run fine-tuning
    trained_model= fine_tune_yolov11n(
                        device="0",
                    )