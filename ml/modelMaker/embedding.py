import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import tensorflow as tf
from mediapipe_model_maker import gesture_recognizer
from PIL import Image

# ===========================
# Config
# ===========================
DATASET_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "datasets", "leapGestRecog_flat"
    )
)
EMBEDDINGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "embeddings"))
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

NUM_WORKERS = os.cpu_count() or 4

# ===========================
# Step 1: Trigger download of TFLite embedder
# ===========================
print("Downloading TFLite gesture embedder using dummy recognizer...")

hparams = gesture_recognizer.HParams(epochs=1, batch_size=1)
options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)

dummy_gr = gesture_recognizer.GestureRecognizer.create(
    train_data=None, validation_data=None, options=options
)

gesture_embedder_path = os.path.join(dummy_gr._model_dir, "gesture_embedder.tflite")
print("Gesture embedder path:", gesture_embedder_path)

if not os.path.exists(gesture_embedder_path):
    raise FileNotFoundError("Gesture embedder TFLite not found after dummy creation!")

# ===========================
# Step 2: Collect all images
# ===========================
image_paths = []
labels = []

for label_folder in os.listdir(DATASET_DIR):
    folder_path = os.path.join(DATASET_DIR, label_folder)
    if not os.path.isdir(folder_path):
        continue
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        image_paths.append(fpath)
        labels.append(label_folder)

print(f"Found {len(image_paths)} images across {len(set(labels))} labels")


# ===========================
# Step 3: Worker function
# ===========================
def process_images_batch(batch_paths, embedder_path):
    interpreter = tf.lite.Interpreter(model_path=embedder_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    batch_embeddings = []
    batch_labels = []

    for img_path in batch_paths:
        try:
            # Use the preprocessed image as-is
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)  # [1,H,W,C]

            interpreter.set_tensor(input_details[0]["index"], img_array)
            interpreter.invoke()
            embedding = interpreter.get_tensor(output_details[0]["index"])[0]

            batch_embeddings.append(embedding)
            batch_labels.append(os.path.basename(os.path.dirname(img_path)))
        except Exception as e:
            print(f"Failed: {img_path} -> {e}")

    return np.array(batch_embeddings), batch_labels


# ===========================
# Step 4: Split list into chunks
# ===========================
def chunk_list(lst, n_chunks):
    k, m = divmod(len(lst), n_chunks)
    return [
        lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n_chunks)
    ]


batches = chunk_list(image_paths, NUM_WORKERS)

# ===========================
# Step 5: Run embedding extraction in parallel
# ===========================
all_embeddings = []
all_labels = []

print(f"Extracting embeddings using {NUM_WORKERS} workers...")

with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = {
        executor.submit(process_images_batch, batch, gesture_embedder_path): i
        for i, batch in enumerate(batches)
    }
    for future in as_completed(futures):
        batch_embeddings, batch_labels = future.result()
        all_embeddings.append(batch_embeddings)
        all_labels.extend(batch_labels)
        total_done = sum(len(e) for e in all_embeddings)
        print(f"Processed batch, total embeddings so far: {total_done}")

# ===========================
# Step 6: Combine and save embeddings
# ===========================
all_embeddings = np.vstack(all_embeddings)
all_labels = np.array(all_labels)

embeddings_file = os.path.join(EMBEDDINGS_DIR, "embeddings.npy")
labels_file = os.path.join(EMBEDDINGS_DIR, "labels.npy")

np.save(embeddings_file, all_embeddings)
np.save(labels_file, all_labels)

print(f"Saved {len(all_embeddings)} embeddings to {embeddings_file}")
print(f"Saved labels to {labels_file}")
