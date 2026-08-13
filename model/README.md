# model/ README

This directory stores model artifacts for the Dog & Cat classifier.

Currently it contains a script to generate a small dummy Keras model for
local testing: generate_dummy_keras_model.py

To create the dummy model locally (requires TensorFlow):

   python -m venv venv
   source venv/bin/activate   # macOS / Linux
   pip install tensorflow==2.14.0
   python model/generate_dummy_keras_model.py

This will create model/model.h5 which the app can load. Replace model.h5
with your real trained model for accurate predictions.
