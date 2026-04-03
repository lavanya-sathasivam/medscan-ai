import os
import shutil
import random
from tqdm import tqdm

input_base = r"C:\Users\lavan\Downloads\brain_ct\data"
output_base = r"C:\Users\lavan\Desktop\MedScan-AI\processed_dataset"
print("Checking input path:", input_base)
print("Exists?", os.path.exists(input_base))
print(os.path.exists(r"C:\Users\lavan\Desktop\MedScan-AI\processed_dataset"))
print("Folders inside:", os.listdir(input_base))
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

classes = {
    "hemorrhagic": "hemorrhage",
    "normal": "normal"
}

for split in ['train', 'val', 'test']:
    for cls in classes.values():
        os.makedirs(os.path.join(output_base, split, cls), exist_ok=True)

data_dict = {cls: [] for cls in classes.values()}

for old_class, new_class in classes.items():
    class_path = os.path.join(input_base, old_class)

    for root, dirs, files in os.walk(class_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(root, file)
                data_dict[new_class].append(full_path)

print("\n📊 Total Images Found:")
for cls in data_dict:
    print(f"{cls}: {len(data_dict[cls])}")

for cls, files in data_dict.items():
    random.shuffle(files)

    total = len(files)
    train_end = int(train_ratio * total)
    val_end = int((train_ratio + val_ratio) * total)

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    def copy_files(file_list, split_name):
        for src in tqdm(file_list, desc=f"{split_name.upper()} - {cls}"):
            filename = os.path.basename(src)
            dst = os.path.join(output_base, split_name, cls, filename)

            # Prevent overwrite
            if os.path.exists(dst):
                filename = f"{os.path.splitext(filename)[0]}_{random.randint(0,9999)}.jpg"
                dst = os.path.join(output_base, split_name, cls, filename)

            shutil.copy(src, dst)

    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')

print("\n✅ FINAL DATASET STRUCTURE:")

for split in ['train', 'val', 'test']:
    print(f"\n{split.upper()}")
    for cls in classes.values():
        path = os.path.join(output_base, split, cls)
        print(f"{cls}: {len(os.listdir(path))}")

print("\nDONE — Your dataset is ready for training.")