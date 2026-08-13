# Dog vs Cat Image Classification - Webapp

This repository contains a Flask web application that provides a simple interface to classify images as Dog or Cat. The current branch (webapp) includes a placeholder prediction function. Replace or add your trained model under the model/ directory and update model/config.json to enable real inference.

Setup
1. Create and activate a virtual environment:

   python -m venv venv
   source venv/bin/activate   # Linux / macOS
   venv\Scripts\activate    # Windows

2. Install dependencies:

   pip install -r requirements.txt

Run

   export FLASK_APP=app.py
   flask run

Or simply:

   python app.py

Open your browser at http://127.0.0.1:5000

Adding a real model

- Place your model file in the model/ directory.
- Edit model/config.json and set the framework, model_path, input_size, color_mode, scale, and class_mapping.

By default, the app uses a simple placeholder classifier so the UI can be tested immediately.
