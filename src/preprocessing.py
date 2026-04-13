import os
import glob
import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN

def run_preprocessing(data_raw_dir, data_processed_dir):
    """
    Main pipeline to run Face Cropping (MTCNN), Depth Map Generation (MiDaS),
    and Meta Lists generation for CASIA, MSU, and NUAA.
    """
    os.makedirs(data_processed_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Init MTCNN for Face Cropping
    mtcnn = MTCNN(select_largest=False, device=device, image_size=256, margin=20)
    
    # 2. Init MiDaS for Depth Map Generation
    print("Loading MiDaS for Depth Map generation...")
    midas = torch.hub.load("isl-org/MiDaS", "MiDaS_small", trust_repo=True).to(device)
    midas.eval()
    midas_transforms = torch.hub.load("isl-org/MiDaS", "transforms", trust_repo=True).small_transform

    datasets = ['CASIA_database', 'MSU-MFSD', 'NUAA']
    dataset_dirs = {'CASIA_database': 'CASIA_faceAntisp', 'MSU-MFSD': 'MSU_MFSD', 'NUAA': 'NUAA Photograph Imposter Database'}

    # Dataset
    # casia: 2 dir train, test. Code mẫu.
    # 
    # msu: train.txt, test.txt
    # nuaa: train.txt, test.txt, ...

    # data/processed/<dataset>/<split>/<label>/<video_prefix>/<frame_id>.jpg
    # if frame_id % 5 == 0: # TODO only 25 frames. Đếm video có n frame, chia đều lấy n/25 frame

    
    for ds_name in datasets:
        raw_ds_path = os.path.join(data_raw_dir, dataset_dirs[ds_name])
        if not os.path.exists(raw_ds_path):
            print(f"Dataset {dataset_dirs[ds_name]} not found in {data_raw_dir}. Skipping...")
            continue
            
        print(f"--- Processing {ds_name} ---")
        save_ds_path = os.path.join(data_processed_dir, ds_name)
        save_depth_path = os.path.join(data_processed_dir, f"{ds_name}_depth")
        os.makedirs(save_ds_path, exist_ok=True)
        os.makedirs(save_depth_path, exist_ok=True)
        
        # We recursively find all video/image files
        files = glob.glob(os.path.join(raw_ds_path, "**", "*.avi"), recursive=True) + \
                glob.glob(os.path.join(raw_ds_path, "**", "*.mp4"), recursive=True) + \
                glob.glob(os.path.join(raw_ds_path, "**", "*.jpg"), recursive=True) + \
                glob.glob(os.path.join(raw_ds_path, "**", "*.png"), recursive=True)
                
        for i, filepath in enumerate(files):
             # Very Basic Real/Fake logic heuristic based on standard folder names
            is_real = ("real" in filepath.lower() or "live" in filepath.lower() or 
                       "client" in filepath.lower() or "1.avi" in filepath.lower() or "2.avi" in filepath.lower() 
                       or "hr_1.avi" in filepath.lower()) # TODO check this
            
            fp = filepath.lower().replace('\\', '/')
            split = 'test'
            if 'train' in fp:
                split = 'train'
            elif 'msu_mfsd' in fp:
                train_subs = [2,3,5,6,7,8,9,11,12,14,21,22,23,24,26,27,28,30,32,33]
                if any(f"/{s:02d}/" in fp or f"/{s:02d}_" in fp for s in train_subs):
                    split = 'train'
            elif 'nuaa' in fp:
                if any(f"/000{s}/" in fp for s in range(1, 10)):
                    split = 'train'
            
            label_str = 'live' if is_real else 'spoof'
            video_prefix = os.path.basename(filepath).split('.')[0]
            
            # Subfolder name format: e.g. casia_train_live_video1
            subfolder_name = f"{ds_name.split('_')[0].lower()}_{split}_{label_str}_{video_prefix}"
            target_subfolder = os.path.join(save_ds_path, subfolder_name)
            target_depth_subfolder = os.path.join(save_depth_path, subfolder_name)
            
            if os.path.exists(target_subfolder) and len(os.listdir(target_subfolder)) > 5:
                continue # Already processed
                
            os.makedirs(target_subfolder, exist_ok=True)
            if is_real:
                os.makedirs(target_depth_subfolder, exist_ok=True)
                
            # Process video or image
            cap = cv2.VideoCapture(filepath)
            frame_id = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                # Sample every 5 frames for videos
                if frame_id % 5 == 0: # TODO only 25 frames. Đếm video có n frame, chia đều lấy n/25 frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    try:
                        # Detect and crop face
                        boxes, _ = mtcnn.detect(frame_rgb)
                        if boxes is not None:
                            # Extract first face
                            cropped_face = mtcnn.extract(frame_rgb, boxes, None)
                            if cropped_face is not None:
                                crop_path = os.path.join(target_subfolder, f"crop_{frame_id:04d}.jpg")
                                # MTCNN returns normalized tensor, convert back to pil
                                cropped_img = ((cropped_face.permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)
                                Image.fromarray(cropped_img).save(crop_path)
                                
                                # Generate Depth if Real
                                if is_real:
                                    depth_path = os.path.join(target_depth_subfolder, f"crop_{frame_id:04d}.jpg")
                                    input_batch = midas_transforms(cropped_img).to(device)
                                    with torch.no_grad():
                                        prediction = midas(input_batch)
                                        prediction = torch.nn.functional.interpolate(
                                            prediction.unsqueeze(1),
                                            size=cropped_img.shape[:2],
                                            mode="bicubic",
                                            align_corners=False,
                                        ).squeeze()
                                    output = prediction.cpu().numpy()
                                    output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                                    cv2.imwrite(depth_path, output)
                    except Exception as e:
                        print(f"Error processing frame {frame_id} from {filepath}: {e}")
                frame_id += 1
            cap.release()
            
            if i % 10 == 0:
                print(f"Processed {i}/{len(files)} files for {ds_name}...")

    # 3. Generate .txt Meta Lists
    print("--- Generating Meta Lists ---")
    list_dir = os.path.join(data_processed_dir, 'meta_lists')
    os.makedirs(list_dir, exist_ok=True)
    
    for ds_name in datasets:
        ds_path = os.path.join(data_processed_dir, ds_name)
        if not os.path.exists(ds_path): continue
        
        prefix = ds_name.split('_')[0].upper()
        
        for split in ['train', 'test']:
            pos_list, neg_list = [], []
            folders = glob.glob(os.path.join(ds_path, f"*_{split}_*"))
            
            for folder in folders:
                images = glob.glob(os.path.join(folder, "crop_*.jpg"))
                # Absolute paths required
                abs_images = [os.path.abspath(img).replace('\\', '/') for img in images]
                
                if '_live_' in folder:
                    pos_list.extend([f"{img} 0" for img in abs_images])
                elif '_spoof_' in folder:
                    neg_list.extend([f"{img} 1" for img in abs_images])
                    
            if pos_list:
                with open(os.path.join(list_dir, f"{prefix}_{split}_pos.txt"), "w") as f:
                    f.write("\n".join(pos_list))
            if neg_list:
                with open(os.path.join(list_dir, f"{prefix}_{split}_neg.txt"), "w") as f:
                    f.write("\n".join(neg_list))
            print(f"Generated {prefix} {split} lists: {len(pos_list)} positive, {len(neg_list)} negative images.")

if __name__ == '__main__':
    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))
    processed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))
    run_preprocessing(raw_dir, processed_dir)
    print("Preprocessing Complete!")
