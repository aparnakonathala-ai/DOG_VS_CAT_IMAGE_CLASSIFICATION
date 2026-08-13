# Dog vs Cat Image Classification - Webapp

This repository contains a Flask web application that provides a simple interface to classify images as Dog or Cat. The current branch (webapp) includes a placeholder prediction function by default.

Keras Integration

- I have integrated a Keras/TensorFlow loader into the Predictor. If you place a Keras model (HDF5 .h5 file or a SavedModel directory) under the model/ directory and set model/config.json to use framework: "keras" (or "tensorflow"), the app will attempt to load it on startup.
- The predictor uses lazy imports and suppresses TensorFlow warnings so the app remains usable even if TensorFlow is not installed.

If you want Keras support locally, install TensorFlow after activating your virtual environment:

   pip install tensorflow==2.14.0

Setup
1. Create and activate a virtual environment:

   python -m venv venv
   source venv/bin/activate   # Linux / macOS
   venv\Scripts\activate    # Windows

2. Install base dependencies:

   pip install -r requirements.txt

3. If you have a Keras model, install TensorFlow as above.

Run

   python app.py

Open your browser at http://127.0.0.1:5000

Adding a real model

- Place your model file in the model/ directory.
  - Keras example: model/model.h5 or model/saved_model/

- Edit model/config.json and set the following fields appropriately:
  - framework: 'keras' | 'pytorch' | 'onnx' | 'placeholder'
  - model_path: relative path to the model file inside model/ (e.g. "model.h5")
  - input_size: [height, width]
  - color_mode: 'rgb' or 'grayscale'
  - scale: e.g. 1.0/255
  - class_mapping: {"0": "Cat", "1": "Dog"}
  - channel_order: 'channels_last' or 'channels_first' (default: channels_last)

Notes on preprocessing and mapping

- The predictor will resize images to input_size and multiply by `scale` if provided.
- For PyTorch models that expect NCHW input, set "channel_order": "channels_first" in config.json.
- The predictor attempts to interpret model outputs as probabilities (sigmoid for single-output, softmax for multi-output). Ensure your model outputs match these conventions or adapt prediction.py accordingly.

If you provide your model here, I can integrate it fully, test, and adjust preprocessing to match the original training pipeline.
