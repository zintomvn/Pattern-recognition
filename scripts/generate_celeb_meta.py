import os

# Configuration
SOURCE_DIR = "data/processed/CelebASpoof"
OUTPUT_DIR = "data/processed/meta_lists"

def generate_lists():
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory {SOURCE_DIR} does not exist.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Categories we are interested in
    sets = ['train', 'test']
    labels = {'live': 0, 'spoof': 1}

    for s in sets:
        for label_name, label_val in labels.items():
            list_filename = f"CelebA_Spoof_{s}_{'pos' if label_val == 1 else 'neg'}.txt"
            list_path = os.path.join(OUTPUT_DIR, list_filename)
            
            entries = []
            # Structure: CelebASpoof/Data/[train|test]/[ID]/[live|spoof]/[FILE]
            set_dir = os.path.join(SOURCE_DIR, 'Data', s)
            if not os.path.exists(set_dir):
                continue
                
            for id_dir in sorted(os.listdir(set_dir)):
                id_path = os.path.join(set_dir, id_dir)
                if not os.path.isdir(id_path):
                    continue
                    
                target_dir = os.path.join(id_path, label_name)
                if not os.path.exists(target_dir):
                    continue
                    
                for file in sorted(os.listdir(target_dir)):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        # Relative path from project root for training script
                        # Example: ../data/processed/CelebASpoof/Data/train/10001/live/000000.jpg 0
                        rel_path = f"../{SOURCE_DIR}/Data/{s}/{id_dir}/{label_name}/{file}"
                        entries.append(f"{rel_path} {label_val}")

            with open(list_path, 'w') as f:
                for entry in entries:
                    f.write(entry + '\n')
            
            print(f"Generated {list_path} with {len(entries)} entries.")

if __name__ == "__main__":
    generate_lists()
