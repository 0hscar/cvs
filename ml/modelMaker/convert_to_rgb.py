import os

from PIL import Image, UnidentifiedImageError

dataset_dir = os.path.join(
    os.path.dirname(__file__), "..", "..", "datasets", "leapGestRecog_flat"
)

count = 0
error_count = 0

for label_folder in os.listdir(dataset_dir):
    label_path = os.path.join(dataset_dir, label_folder)
    if not os.path.isdir(label_path):
        continue
    for img_file in os.listdir(label_path):
        img_path = os.path.join(label_path, img_file)
        try:
            with Image.open(img_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                    img.save(img_path)
            count += 1
            if count % 100 == 0:
                print(f"Processed {count} images...")
        except UnidentifiedImageError:
            print(f"Skipping non-image file: {img_path}")
            error_count += 1
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
            error_count += 1

print(f"Done! Processed {count} images with {error_count} errors/skipped files.")
