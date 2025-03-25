import os
import shutil
import scipy.io
from PIL import Image

# Paths
anno_file = 'Dataset/cars_annos.mat'
output_dir = 'yolo_dataset'

# Make output folders
for split in ['train', 'val']:
    os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

# Load annotation file
mat = scipy.io.loadmat(anno_file)
annotations = mat['annotations'][0]

for a in annotations:
    original_filename = a[0][0]  # e.g. 'cars_train/000001.jpg'
    x1, y1, x2, y2 = a[1][0][0], a[2][0][0], a[3][0][0], a[4][0][0]
    class_id = a[5][0][0] - 1  # convert to 0-indexed
    is_test = a[6][0][0]

    # Fix filename: annotation uses 6-digit, your files use 5-digit
    # Extract number from filename and rebuild
    img_number = int(original_filename.split('/')[-1].split('.')[0])  # get the number only
    fixed_name = f"{img_number:05d}.jpg"  # make it 5-digit format

    # Choose source path based on train/test
    if is_test:
        img_path = os.path.join('Dataset', 'cars_test', fixed_name)
        split = 'val'
    else:
        img_path = os.path.join('Dataset', 'cars_train', fixed_name)
        split = 'train'

    # Skip if file doesn't exist
    if not os.path.exists(img_path):
        print(f"⚠️ File not found: {img_path}, skipping...")
        continue

    # Load image to get width/height
    img = Image.open(img_path)
    w, h = img.size

    # Convert to YOLO format
    x_center = ((x1 + x2) / 2.0) / w
    y_center = ((y1 + y2) / 2.0) / h
    bbox_width = (x2 - x1) / w
    bbox_height = (y2 - y1) / h

    yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"

    # Copy image to correct folder
    dest_img_path = os.path.join(output_dir, 'images', split, fixed_name)
    shutil.copy(img_path, dest_img_path)

    # Save label file
    label_filename = fixed_name.replace('.jpg', '.txt')
    label_path = os.path.join(output_dir, 'labels', split, label_filename)
    with open(label_path, 'w') as f:
        f.write(yolo_line + '\n')

print("✅ Dataset successfully converted to YOLO format!")
