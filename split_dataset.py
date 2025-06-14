import os
import shutil
import random

def split_data(source, train, val, test, split_ratio=(0.7, 0.2, 0.1)):
    files = os.listdir(source)
    files = [f for f in files if os.path.isfile(os.path.join(source, f))]
    random.shuffle(files)

    train_split = int(len(files) * split_ratio[0])
    val_split = train_split + int(len(files) * split_ratio[1])

    train_files = files[:train_split]
    val_files = files[train_split:val_split]
    test_files = files[val_split:]

    for file in train_files:
        shutil.copy(os.path.join(source, file), os.path.join(train, file))
    for file in val_files:
        shutil.copy(os.path.join(source, file), os.path.join(val, file))
    for file in test_files:
        shutil.copy(os.path.join(source, file), os.path.join(test, file))

base_dir = 'Dataset'
output_dir = 'dataset'
os.makedirs(output_dir, exist_ok=True)

for category in ['NORMAL', 'PNEUMONIA']:
    os.makedirs(f'{output_dir}/train/{category}', exist_ok=True)
    os.makedirs(f'{output_dir}/val/{category}', exist_ok=True)
    os.makedirs(f'{output_dir}/test/{category}', exist_ok=True)

    split_data(f'{base_dir}/{category}',
               f'{output_dir}/train/{category}',
               f'{output_dir}/val/{category}',
               f'{output_dir}/test/{category}')
print("✅ Dataset split completed.")
