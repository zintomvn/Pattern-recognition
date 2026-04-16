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
SAMPLES_PER_CLASS = 100

def get_samples(label_path):
    live_samples = []
    spoof_samples = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            img_path = parts[0]
            label = int(parts[1])
            # 0 is Live (Neg in anti-spoofing), 1 is Spoof (Pos)
            if label == 0:
                live_samples.append((img_path, label))
            elif label == 1:
                spoof_samples.append((img_path, label))
                
    selected_live = random.sample(live_samples, min(len(live_samples), SAMPLES_PER_CLASS))
    selected_spoof = random.sample(spoof_samples, min(len(spoof_samples), SAMPLES_PER_CLASS))
    return selected_live, selected_spoof

def download_file(img_path, temp_dir):
    target_path = os.path.join(OUTPUT_ROOT, img_path)
    if os.path.exists(target_path):
        return True
        
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    # In this dataset, files are prefixed with CelebA_Spoof/ on Kaggle
    kaggle_path = f"CelebA_Spoof/{img_path}"
    
    # Use Kaggle CLI which correctly authenticates and fetches single files via -f
    cmd = [
        "kaggle", "datasets", "download", "-d", DATASET_HANDLE,
        "-f", kaggle_path, "-p", temp_dir
    ]
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
                # We want output format: ../data/raw/CelebASpoof/Data/train/... 1
                f.write(f"../{OUTPUT_ROOT}/{img_path} {label}\n")
                count += 1
    print(f"Exported {count} items to {filepath}")

def process_set(label_path, set_name):
    print(f"--- Processing {set_name} set ---")
    live_samples, spoof_samples = get_samples(label_path)
    
    all_images = [img for img, _ in live_samples] + [img for img, _ in spoof_samples]
    
    temp_dir = f"./tmp_dl_kaggle_{set_name}"
    os.makedirs(temp_dir, exist_ok=True)
    
    print(f"Downloading {len(all_images)} images over multiple threads...")
    success_count = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(download_file, img, temp_dir): img for img in all_images}
        for future in as_completed(futures):
            if future.result():
                success_count += 1

    print(f"Downloaded {success_count}/{len(all_images)} images successfully.")
    
    # Generate list files in requested format
    export_list(live_samples, f"CelebA_Spoof_{set_name}_neg.txt")
    export_list(spoof_samples, f"CelebA_Spoof_{set_name}_pos.txt")
    
    # cleanup temp dir
    shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    random.seed(42)
    
    # Read Kaggle token if exists so we can use KGAT_ API token instead of username/key combo
    token_file = "kaggle_token.txt"
    if os.path.exists(token_file):
        with open(token_file, 'r') as f:
            os.environ['KAGGLE_KEY'] = f.read().strip()
    
    process_set(TRAIN_LABEL_PATH, "train")
    process_set(TEST_LABEL_PATH, "test")
    
    print("Downloading and Exporting Complete!")

if __name__ == "__main__":
    main()
