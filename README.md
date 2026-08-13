# Dog vs Cat Image Classification - Webapp

This repository contains a Flask web application that provides a simple interface to classify images as Dog or Cat. The current branch (webapp) includes a placeholder prediction function by default.

If you have a trained model, place it under the model/ directory and update model/config.json. The app supports (via lazy imports) the following frameworks:
- Keras / TensorFlow (.h5 or SavedModel)
- PyTorch (.pt / .pth)
- ONNX (.onnx)

Setup
1. Create and activate a virtual environment:

   python -m venv venv
   source venv/bin/activate   # Linux / macOS
   venv\Scripts\activate    # Windows

2. Install base dependencies:

   pip install -r requirements.txt

3. If you plan to use a real model, uncomment the appropriate framework in requirements.txt and install it. Examples:

   # For Keras/TensorFlow
   pip install tensorflow==2.14.0

   # For PyTorch
   pip install torch

   # For ONNX Runtime
   pip install onnxruntime

Run

   python app.py

Open your browser at http://127.0.0.1:5000

Adding a real model

- Place your model file in the model/ directory.
  - Keras example: model/model.h5
  - PyTorch example: model/model.pt
  - ONNX example: model/model.onnx

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

If you provide your model here, I can integrate it, test, and adjust preprocessing to match the original training pipeline.
