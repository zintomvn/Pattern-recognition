import os
import random
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# Local configurations
TRAIN_LABEL_PATH = "/home/nesfan/Desktop/HCMUS/Nam3/HK2/NhanDang/train_label/train_label.txt"
TEST_LABEL_PATH = "/home/nesfan/Desktop/HCMUS/Nam3/HK2/NhanDang/train_label/test_label.txt"

DATASET_HANDLE = "mabdullahsajid/celeba-spoofing"
OUTPUT_ROOT = "data/raw/CelebASpoof"
META_LISTS_DIR = "data/processed/meta_lists_small_basic"
TARGET_IDENTITIES = 100

def get_samples(label_path):
    """
    Groups images by identity and selects the first 100 identities
    that have both at least one live and one spoof image.
    Returns: (selected_live, selected_spoof) where each is a list of (path, label)
    """
    identities = {} # {id_str: {'live': path, 'spoof': path}}
    ordered_ids = []
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            img_path = parts[0]
            label = int(parts[1])
            
            # Extract ID: Data/train/ID/live/000000.jpg -> components[2]
            components = img_path.split('/')
            if len(components) < 3:
                continue
            identity = components[2]
            
            if identity not in identities:
                identities[identity] = {'live': None, 'spoof': None}
                ordered_ids.append(identity)
                
            # Pick the first one found (user request)
            if label == 0 and identities[identity]['live'] is None:
                identities[identity]['live'] = img_path
            elif label == 1 and identities[identity]['spoof'] is None:
                identities[identity]['spoof'] = img_path
    
    selected_live = []
    selected_spoof = []
    count = 0
    
    for identity in ordered_ids:
        data = identities[identity]
        if data['live'] and data['spoof']:
            selected_live.append((data['live'], 0))
            selected_spoof.append((data['spoof'], 1))
            count += 1
            if count >= TARGET_IDENTITIES:
                break
                
    return selected_live, selected_spoof

def download_file(img_path, temp_dir):
    target_path = os.path.join(OUTPUT_ROOT, img_path)
    if os.path.exists(target_path):
        return True
        
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    kaggle_path = f"CelebA_Spoof/{img_path}"
    
    cmd = [
        "kaggle", "datasets", "download", "-d", DATASET_HANDLE,
        "-f", kaggle_path, "-p", temp_dir
    ]
    # No shell=True to avoid injection; subprocess Handles list correctly
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    filename = os.path.basename(kaggle_path)
    dl_file = os.path.join(temp_dir, filename)
    
    if os.path.exists(dl_file):
        shutil.move(dl_file, target_path)
        print(f"Downloaded {img_path}")
        return True
    else:
        print(f"Failed to download {kaggle_path}")
        return False

def export_list(samples, filename):
    os.makedirs(META_LISTS_DIR, exist_ok=True)
    filepath = os.path.join(META_LISTS_DIR, filename)
    count = 0
    with open(filepath, 'w') as f:
        for img_path, label in samples:
            target_path = os.path.join(OUTPUT_ROOT, img_path)
            if os.path.exists(target_path):
                f.write(f"../{OUTPUT_ROOT}/{img_path} {label}\n")
                count += 1
    print(f"Exported {count} items to {filepath}")

def process_set(label_path, set_name):
    print(f"--- Processing {set_name} set (Identity-based) ---")
    live_samples, spoof_samples = get_samples(label_path)
    
    all_images = [img for img, _ in live_samples] + [img for img, _ in spoof_samples]
    
    print(f"Total identities selected: {len(live_samples)}")
    print(f"Total images to ensure: {len(all_images)}")

    temp_dir = f"./tmp_dl_kaggle_{set_name}"
    os.makedirs(temp_dir, exist_ok=True)
    
    success_count = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(download_file, img, temp_dir): img for img in all_images}
        for future in as_completed(futures):
            if future.result():
                success_count += 1

    print(f"Downloaded/Verified {success_count}/{len(all_images)} images.")
    
    export_list(live_samples, f"CelebA_Spoof_{set_name}_neg.txt")
    export_list(spoof_samples, f"CelebA_Spoof_{set_name}_pos.txt")
    
    shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    # Read Kaggle token if exists
    token_file = "kaggle_token.txt"
    if os.path.exists(token_file):
        with open(token_file, 'r') as f:
            os.environ['KAGGLE_KEY'] = f.read().strip()
    
    process_set(TRAIN_LABEL_PATH, "train")
    process_set(TEST_LABEL_PATH, "test")
    
    print("Identity-based Downloading and Exporting Complete!")

if __name__ == "__main__":
    main()
