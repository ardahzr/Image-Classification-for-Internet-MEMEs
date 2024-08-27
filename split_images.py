import os
import random
import shutil

def split_images(source_dir, train_dir, val_dir, test_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):

    
    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    # Create the output directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get all image files in the source directory
    image_files = [
        f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))
    ]

    # Shuffle the image files randomly
    random.shuffle(image_files)

    # Calculate the split indices
    total_images = len(image_files)
    train_split = int(total_images * train_ratio)
    val_split = int(total_images * (train_ratio + val_ratio))

    # Split the images into train, validation, and test sets
    train_images = image_files[:train_split]
    val_images = image_files[train_split:val_split]
    test_images = image_files[val_split:]

    # Copy images to their respective directories
    for image in train_images:
        shutil.copy(os.path.join(source_dir, image), os.path.join(train_dir, image))
    for image in val_images:
        shutil.copy(os.path.join(source_dir, image), os.path.join(val_dir, image))
    for image in test_images:
        shutil.copy(os.path.join(source_dir, image), os.path.join(test_dir, image))

    print(f"Split {total_images} images into:")
    print(f"  Training: {len(train_images)} images in {train_dir}")
    print(f"  Validation: {len(val_images)} images in {val_dir}")
    print(f"  Testing: {len(test_images)} images in {test_dir}")



source_directory = "/content/archive/Files"  # Replace with your actual directory
train_directory = "/content/archive/train"
validation_directory = "/content/archive/val"
test_directory = "/content/archive/test"

split_images(source_directory, train_directory, validation_directory, test_directory)