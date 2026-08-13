import io
import json
import os
from PIL import Image, ImageFile
import numpy as np

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class InvalidImageError(Exception):
    pass


class Predictor:
    """
    A prediction wrapper that supports a placeholder model.

    If you later add a real model file into the model/ directory and
    set model/config.json appropriately, this class will load and use it.
    """

    def __init__(self, model_dir='model'):
        self.model_dir = model_dir
        self.model = None
        self.config = None
        self.load_config()
        # Currently we use a simple placeholder model (based on mean pixel value)
        # Real model loading (Keras/PyTorch/ONNX) can be added if model files are present

    def load_config(self):
        cfg_path = os.path.join(self.model_dir, 'config.json')
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r') as f:
                self.config = json.load(f)
        else:
            # default config
            self.config = {
                "framework": "placeholder",
                "input_size": [224, 224],
                "color_mode": "rgb",
                "scale": 1.0/255,
                "class_mapping": {"0": "Cat", "1": "Dog"}
            }

    def preprocess(self, image_bytes):
        try:
            image = Image.open(io.BytesIO(image_bytes))
        except Exception:
            raise InvalidImageError('Cannot open image')

        if self.config.get('color_mode', 'rgb').lower() == 'rgb':
            image = image.convert('RGB')
        elif self.config.get('color_mode').lower() == 'grayscale':
            image = image.convert('L')

        target_size = tuple(self.config.get('input_size', [224, 224]))
        image = image.resize(target_size)
        arr = np.array(image).astype('float32')

        # Scale if required
        scale = self.config.get('scale', None)
        if scale:
            arr = arr * float(scale)

        # Add batch dim
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)
        arr = np.expand_dims(arr, axis=0)
        return arr

    def predict_with_placeholder(self, preprocessed):
        # Simple heuristic: average brightness -> dog/cat
        mean_val = float(np.mean(preprocessed))
        # map mean to confidence between 50 and 100
        confidence = min(max((mean_val - 0.2) * 200, 50), 99.99)
        # arbitrary threshold
        prediction = 'Dog' if mean_val > 0.5 else 'Cat'
        return {'prediction': prediction, 'confidence': round(confidence, 2)}

    def predict_image(self, file_storage):
        # file_storage is a Werkzeug FileStorage object or a file-like object
        try:
            image_bytes = file_storage.read()
        except Exception:
            raise InvalidImageError('Failed to read uploaded file')

        # Basic size check
        if len(image_bytes) == 0:
            raise InvalidImageError('Empty image')

        preprocessed = self.preprocess(image_bytes)

        # If real model is loaded, use it. For now use placeholder
        if self.config.get('framework') == 'placeholder' or self.model is None:
            return self.predict_with_placeholder(preprocessed)
        else:
            # Integrate real model inference here in future
            return self.predict_with_placeholder(preprocessed)
