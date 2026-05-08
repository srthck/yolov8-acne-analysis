import os
from ultralytics import YOLO

def main():
    # Define project paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(project_root, 'data.yaml')
    
    # Verify the yaml file exists
    if not os.path.exists(yaml_path):
        print(f"Error: Could not find dataset config at {yaml_path}")
        return

    # Initialize the YOLOv8 Segmentation model
    # Using the 'small' model as it offers a good balance for detailed features like acne
    model = YOLO('yolov8s-seg.pt')

    print("Starting YOLOv8 Segmentation training pipeline...")
    
    # Train the model
    # Parameters configured for stability and standard hardware capability
    results = model.train(
        data=yaml_path,
        epochs=50,             # Configurable: Reduce for testing, increase for final model
        imgsz=416,             # Reduced from 640 for CPU memory (still captures lesion detail)
        batch=4,               # Reduced from 16 for CPU (major memory constraint)
        patience=10,           # Early stopping patience
        project='outputs',     # Save results in the outputs directory
        name='acne_segmentation',
        device='',             # Auto-detect device (CPU/GPU)
        exist_ok=True,         # Overwrite previous runs with the same name if needed
        workers=0,             # Disable multiprocessing on CPU to save memory
        # ── Augmentation overrides ───────────────────────────────────────────
        mosaic=0.0,            # DISABLED: mosaic merges 4 face crops into 1 tile → invalid anatomy
        degrees=10.0,          # Mild rotation ±10° is safe and realistic for face photos
        fliplr=0.5,            # Horizontal flip: valid (faces are symmetric)
        perspective=0.0,       # Disable perspective distortion to reduce memory during augmentation
        erasing=0.0,           # Disable random erasing to reduce memory
    )
    
    print("Training complete! Model saved to outputs/acne_segmentation/")

if __name__ == '__main__':
    main()
