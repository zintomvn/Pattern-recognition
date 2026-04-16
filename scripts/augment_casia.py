import os
import sys
import cv2
import random
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as tv_transforms
import albumentations as A

# Constants
META_LIST_PATH = "data/processed/meta_lists/CASIA_train_pos.txt"
OUTPUT_DIR_BASE = "data/processed/CASIA/train/spoof"
FILE_LIMIT = 100
INPUT_SIZE = 256

# Add src to path to import Moire
sys.path.append(os.path.join(os.getcwd(), 'src'))
from extensions.simulate.moire import Moire

class ColorTrans(object):
    """Transpose the image color space from RGB to BGR or reverse.
    Args:
        mode (int): 0 for BGR to RGB, 1 for RGB to BGR.
    """
    def __init__(self, mode=0):
        self.mode = mode

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor/ndarray): Image of size (H, W, C) to be converted.
        Returns:
            ndarray: Converted image.
        """
        # Note: input is expected as ndarray for cv2.cvtColor
        if isinstance(tensor, Image.Image):
            tensor = np.array(tensor)
            
        if self.mode == 0 and tensor.shape[2] == 3:
            return cv2.cvtColor(tensor, cv2.COLOR_BGR2RGB)
        elif self.mode == 1 and tensor.shape[2] == 3:
            return cv2.cvtColor(tensor, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError("Input should be a 3 channel images.")

    def __repr__(self):
        return self.__class__.__name__ + '(mode={0})'.format(self.mode)


def augment_image(image, moire_gen):
    prob_value = random.random()
    
    # 1. Moire or Color Jitter (Original logic)
    if prob_value < 0.5:
        image = moire_gen(image)
    else:
        color_jitter = tv_transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        transformed_image_pil = color_jitter(image_pil)
        transformed_image_np = np.array(transformed_image_pil)
        image = cv2.cvtColor(transformed_image_np, cv2.COLOR_RGB2BGR)

    # 2. transforms1 (Geometric & ColorTrans)
    t1 = tv_transforms.Compose([
        tv_transforms.RandomResizedCrop(size=INPUT_SIZE, scale=(0.75, 1.0), ratio=(3.0/4.0, 4.0/3.0)),
        ColorTrans(mode=0), # BGR to RGB
    ])
    
    img_pilot = Image.fromarray(image) # Treat current image as BGR-in-RGB-memory for ColorTrans
    img_rgb = t1(img_pilot)
    
    # 3. transforms2 (Albumentations)
    t2 = A.Compose([
        A.HorizontalFlip(p=0.5),
    ])
    
    augmented_dict = t2(image=img_rgb)
    img_final_rgb = augmented_dict['image']
    
    # Convert back to viewable BGR for JPG saving
    img_viewable_bgr = cv2.cvtColor(img_final_rgb, cv2.COLOR_RGB2BGR)
    
    return img_viewable_bgr

def resolve_path(relative_path):
    clean_path = relative_path.lstrip('./').lstrip('../')
    if os.path.exists(clean_path):
        return clean_path
    if "CASIA/" in clean_path:
        db_path = clean_path.replace("CASIA/", "CASIA_database/")
        if os.path.exists(db_path):
            return db_path
    return None

def main():
    if not os.path.exists(META_LIST_PATH):
        print(f"Error: Meta list not found at {META_LIST_PATH}")
        return

    moire = Moire()
    os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)
    
    processed_count = 0
    with open(META_LIST_PATH, 'r') as f:
        for line in f:
            if processed_count >= FILE_LIMIT:
                break
                
            parts = line.strip().split()
            if not parts:
                continue
            
            img_rel_path = parts[0]
            actual_path = resolve_path(img_rel_path)
            if not actual_path:
                continue
                
            img = cv2.imread(actual_path)
            if img is None:
                continue
            
            # Augment
            try:
                augmented_img = augment_image(img, moire)
                
                # Output filename
                base_name = os.path.basename(img_rel_path)
                name, ext = os.path.splitext(base_name)
                output_name = f"{name}_augment.jpg"
                output_path = os.path.join(OUTPUT_DIR_BASE, output_name)
                
                # Save
                cv2.imwrite(output_path, augmented_img)
                processed_count += 1
                print(f"[{processed_count}] Augmenting: {base_name} -> {output_path}")
            except Exception as e:
                print(f"Error augmenting {img_rel_path}: {e}")

    print(f"Finished. Successfully augmented {processed_count} images.")

if __name__ == "__main__":
    main()
