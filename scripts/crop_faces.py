import os
import cv2
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Initialize OpenCV DNN Face Detector
# Using Haar cascade (built-in, reliable for this task)
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

if face_detector.empty():
    logging.error("Failed to load face detection model. Exiting.")
    exit(1)

def get_padded_bbox(bbox, img_width, img_height, padding_ratio=0.3):
    """
    Calculates a padded bounding box around the face, ensuring it stays within image boundaries.
    """
    xmin, ymin, width, height = bbox
    
    # Calculate padding pixels based on face width/height
    pad_w = int(width * padding_ratio)
    pad_h = int(height * padding_ratio)
    
    # Apply padding
    new_xmin = xmin - pad_w
    new_ymin = ymin - pad_h
    new_xmax = xmin + width + pad_w
    new_ymax = ymin + height + pad_h
    
    # Boundary constraints to prevent exceeding image limits
    new_xmin = max(0, new_xmin)
    new_ymin = max(0, new_ymin)
    new_xmax = min(img_width, new_xmax)
    new_ymax = min(img_height, new_ymax)
    
    return new_xmin, new_ymin, new_xmax, new_ymax

def process_images(input_dir, output_dir):
    """
    Detects faces in images, crops them with padding, and saves to the output directory.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        logging.error(f"Input directory does not exist: {input_dir}")
        return
        
    output_path.mkdir(parents=True, exist_ok=True)
    
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in valid_extensions]
    
    if not image_files:
        logging.warning(f"No valid images found in {input_dir}")
        return

    logging.info(f"Found {len(image_files)} images. Starting face cropping pipeline...")
    
    success_count = 0
    for img_file in image_files:
        try:
            image = cv2.imread(str(img_file))
            if image is None:
                logging.warning(f"Skipping {img_file.name}: Could not read image.")
                continue
                
            img_height, img_width, _ = image.shape
            
            # Detect faces using Haar cascade with preprocessing
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Enhance image contrast for better detection
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)
            
            # Try detection with multiple parameter sets for robustness
            detections = face_detector.detectMultiScale(
                enhanced_gray,
                scaleFactor=1.03,        # Very sensitive scale detection
                minNeighbors=2,          # Very relaxed
                minSize=(20, 20)         # Allow very small faces
            )
            
            # If no detections, try on original image with different params
            if len(detections) == 0:
                detections = face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.05,
                    minNeighbors=2,
                    minSize=(25, 25)
                )
            
            if len(detections) == 0:
                logging.info(f"No face detected in {img_file.name}. Skipping.")
                continue
            
            # Get the largest detected face (typically the main subject)
            detection = max(detections, key=lambda x: x[2] * x[3])
            xmin, ymin, width, height = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3])
            
            # Get padded bounding box
            pxmin, pymin, pxmax, pymax = get_padded_bbox(
                (xmin, ymin, width, height), 
                img_width, 
                img_height, 
                padding_ratio=0.3
            )
            
            # Ensure valid crop region
            if pxmax <= pxmin or pymax <= pymin:
                logging.warning(f"Invalid crop dimensions for {img_file.name}. Skipping.")
                continue
                
            # Crop the image
            cropped_face = image[pymin:pymax, pxmin:pxmax]
            
            if cropped_face.size == 0:
                logging.warning(f"Empty crop for {img_file.name}. Skipping.")
                continue
                
            # Save the cropped face
            out_file_path = output_path / f"cropped_{img_file.name}"
            cv2.imwrite(str(out_file_path), cropped_face)
            success_count += 1
            
        except Exception as e:
            logging.error(f"Error processing {img_file.name}: {e}")
            continue
            
    logging.info(f"Pipeline complete. Successfully cropped {success_count}/{len(image_files)} images.")

if __name__ == "__main__":
    # Define directories relative to project root
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
    CROPPED_DIR = PROJECT_ROOT / "dataset" / "cropped_raw"
    
    process_images(RAW_DIR, CROPPED_DIR)
