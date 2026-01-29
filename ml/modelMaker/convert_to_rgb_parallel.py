import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from PIL import Image, UnidentifiedImageError

# Dataset path
dataset_dir = os.path.join(
    os.path.dirname(__file__), "..", "..", "datasets", "leapGestRecog_flat"
)

# Target size
TARGET_SIZE = (256, 256)  # Resize all images to 256x256


def convert_image(img_path):
    try:
        with Image.open(img_path) as img:
            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Resize image
            img = img.resize(TARGET_SIZE, resample=Image.LANCZOS)
            img.save(img_path)

        return (img_path, True, None)
    except UnidentifiedImageError:
        return (img_path, False, "UnidentifiedImageError")
    except Exception as e:
        return (img_path, False, str(e))


# Gather all image paths
image_paths = []
for label_folder in os.listdir(dataset_dir):
    label_path = os.path.join(dataset_dir, label_folder)
    if not os.path.isdir(label_path):
        continue
    for img_file in os.listdir(label_path):
        img_path = os.path.join(label_path, img_file)
        image_paths.append(img_path)

num_workers = os.cpu_count() or 4
print(f"Processing {len(image_paths)} images using {num_workers} cores...")

# Multiprocessing
success_count = 0
error_count = 0
with ProcessPoolExecutor(max_workers=num_workers) as executor:
    futures = {executor.submit(convert_image, path): path for path in image_paths}
    for i, future in enumerate(as_completed(futures), 1):
        img_path, success, error = future.result()
        if success:
            success_count += 1
        else:
            error_count += 1
            print(f"Error processing {img_path}: {error}")
        if i % 100 == 0:
            print(f"Processed {i} images...")

print(
    f"Done! Successfully processed {success_count} images, {error_count} errors/skipped files."
)
