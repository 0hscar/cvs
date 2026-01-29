import datetime
import os

# Wayland (Arch) + Nvidia + Cuda + Tensorflow + mediapipe says F**k you, no GPU for you.
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
import tensorflow as tf
from mediapipe_model_maker import gesture_recognizer

print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))
print("GPUs:", tf.config.list_physical_devices("GPU"))

# TensorBoard callback
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
)

# Test set
# DST_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "testSet")
# Dataset
DST_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "..", "datasets", "leapGestRecog_flat"
)
data = gesture_recognizer.Dataset.from_folder(
    dirname=DST_ROOT, hparams=gesture_recognizer.HandDataPreprocessingParams()
)

train_data, rest_data = data.split(0.8)
validation_data, test_data = rest_data.split(0.5)


print("Train examples:", train_data)
print("Validation examples:", validation_data)
print("Test examples:", test_data)
timestamp = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
export_dir = os.path.abspath(
    f"/home/ossi/code/cvs/models/exported/gesture_model_{timestamp}"
)

hparams = gesture_recognizer.HParams(
    epochs=40,
    batch_size=16,
    export_dir=export_dir,
)
options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)

# Create + train model
model = gesture_recognizer.GestureRecognizer.create(
    train_data=train_data,
    validation_data=validation_data,
    options=options,
)

# Evaluate
model.evaluate(test_data)

# Export TFLite model
model.export_model()
