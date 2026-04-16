import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
from concurrent.futures.thread import ThreadPoolExecutor

# Initialization
app = FaceAnalysis(name='buffalo_s', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(256, 256))
delta = 20  # expand 20 pixels around bbox

def crop_face_func(img_full_path, output_path):
    """Detects first face, expands bbox, crops, and saves to output_path."""
    img = cv2.imread(img_full_path)
    if img is None:
        print(f"Error: Could not read {img_full_path}")
        return
        
    height, width, _ = img.shape
   

    try:
        faces = app.get(img)
        if not faces:
            print(f"No face detected in: {img_full_path}")
            return

        if height < 300 and width < 300:
            print(f"Skipping small image: {img_full_path} ({width}x{height})")
            crop_face = img
        else:
            bbox = faces[0]['bbox']
            point1 = [int(bbox[0]), int(bbox[1])]
            point2 = [int(bbox[2]), int(bbox[3])]

            # Expand bbox
            point_1 = [max(0, point1[0] - delta), max(0, point1[1] - delta)]
            point_2 = [min(point2[0] + delta, width), min(point2[1] + delta, height)]
            
            crop_face = img[point_1[1]:point_2[1], point_1[0]:point_2[0], :]

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, crop_face)
        print(f"Processed: {os.path.basename(img_full_path)}")
    except Exception as e:
        print(f"Error processing {img_full_path}: {e}")

def process_recursively(input_root, output_root, executor):
    """Recursively traverses input_root and submits tasks to executor."""
    supported_exts = ('.jpg', '.png', '.jpeg')
    cnt = 0
    
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(supported_exts):
                input_path = os.path.join(root, file)
                
                # Maintain relative structure
                rel_path = os.path.relpath(input_path, input_root)
                output_path = os.path.join(output_root, rel_path)
                
                cnt += 1
                executor.submit(crop_face_func, input_path, output_path)
    
    print(f"Submitted {cnt} images for processing.")

if __name__ == '__main__':
    # Configuration
    INPUT_DIR = "data/raw/CelebASpoof"
    OUTPUT_DIR = "data/processed/CelebASpoof"
    MAX_WORKERS = 32

    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory {INPUT_DIR} does not exist.")
    else:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            process_recursively(INPUT_DIR, OUTPUT_DIR, executor)

    print("Finished submitting all tasks.")
