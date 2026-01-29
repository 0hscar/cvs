# CVS (Computer Vision System)

A small computer vision project for hand and gesture recognition.

This repository contains:
- `app/` — application code and camera integration
  - Camera capture, gesture actions, and hand recognition logic
  - `app/main.py` — entry point for running the demo application
  - `app/requirements.txt` — dependencies for running the app
- `ml/` — scripts and utilities for preparing data and training models
  - `ml/modelMaker/` — model preparation, conversion, and embedding utilities
  - Training and embedding helper scripts used to create the exported models
- `models/` — exported trained model artifacts and checkpoints
  - Contains exported model folders and saved weights/checkpoints
- `datasets/` — (optional) datasets used for training and evaluation
- `.gitignore` — repository ignore rules

Quick start
1. Create and activate a Python virtual environment (recommended)
   - python3 -m venv .venv
   - source .venv/bin/activate
2. Install dependencies for the app
   - pip install -r app/requirements.txt
3. Run the demo application (ensure a camera is connected)
   - python app/main.py

Training / model preparation
- Use the scripts under `ml/modelMaker/` to prepare data, generate embeddings, and train or convert models.
- Check the individual script headers for usage examples and arguments (e.g., `ml/modelMaker/main.py`, `convert_to_rgb.py`, `embedding.py`).

Models
- Exported model artifacts and checkpoints live under `models/exported/`.
- Use these artifacts directly with the inference code in `app/` or retrain/convert using the `ml/` tools.

Notes
- This README provides an overview. For detailed usage or development notes, check inline comments in the scripts under `app/` and `ml/`.
- Adjust camera index or configuration in `app/camera/cameraCapture.py` if you have multiple cameras.

License
- No license specified. Add a `LICENSE` file if you intend to open-source this project.
