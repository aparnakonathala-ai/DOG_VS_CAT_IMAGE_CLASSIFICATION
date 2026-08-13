"""Generate and save a tiny dummy Keras model for local testing.

This script creates a simple convolutional model and saves it as model.h5
in the same directory. Install TensorFlow in your venv before running:

    pip install tensorflow==2.14.0

Then run:

    python generate_dummy_keras_model.py

This is intended only for testing the webapp; the model is untrained and
produces meaningless predictions. Replace with your real trained model
for actual inference.
"""

import os

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
except Exception as e:
    print('TensorFlow is not installed. Install it with `pip install tensorflow` before running this script.')
    raise

MODEL_DIR = os.path.dirname(__file__)
OUTPUT_PATH = os.path.join(MODEL_DIR, 'model.h5')

def create_and_save_model(path):
    input_shape = (224, 224, 3)
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Rescaling(1./255),
        layers.Conv2D(8, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.save(path)
    print(f'Dummy Keras model saved to: {path}')

if __name__ == '__main__':
    os.makedirs(MODEL_DIR, exist_ok=True)
    create_and_save_model(OUTPUT_PATH)
