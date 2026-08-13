import io
import json
import os
import logging
from PIL import Image, ImageFile
import numpy as np

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class InvalidImageError(Exception):
    pass


class Predictor:
    """
    A prediction wrapper that supports multiple frameworks via lazy imports.

    Behavior:
    - Loads model/config.json if present in model_dir.
    - Attempts to load the model for the specified framework ('keras', 'pytorch', 'onnx').
    - If loading fails (missing package or model), falls back to a placeholder heuristic so the app remains runnable.

    Notes:
    - Heavy ML packages are imported only if needed (lazy import). This avoids forcing users to install TensorFlow/PyTorch unless they intend to use them.
    - Update model/config.json with correct fields for your model (framework, model_path, input_size, color_mode, scale, class_mapping, channel_order).
    """

    def __init__(self, model_dir='model'):
        self.model_dir = model_dir
        self.model = None
        self.config = None
        self.framework = None
        self._loaded = False
        self.load_config()
        self._try_load_model()

    def load_config(self):
        cfg_path = os.path.join(self.model_dir, 'config.json')
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r') as f:
                self.config = json.load(f)
        else:
            # default config
            self.config = {
                "framework": "placeholder",
                "model_path": "model/model.h5",
                "input_size": [224, 224],
                "color_mode": "rgb",
                "scale": 1.0/255,
                "class_mapping": {"0": "Cat", "1": "Dog"},
                "channel_order": "channels_last"
            }

    def _try_load_model(self):
        fw = self.config.get('framework', 'placeholder').lower()
        model_path = os.path.join(self.model_dir, os.path.basename(self.config.get('model_path', '')))
        self.framework = fw

        if fw == 'placeholder':
            logging.info('Using placeholder predictor (no model).')
            return

        if not os.path.exists(model_path):
            logging.warning('Model file not found at %s — falling back to placeholder.', model_path)
            return

        try:
            if fw == 'keras' or fw == 'tensorflow':
                # Lazy import tensorflow.keras
                try:
                    from tensorflow.keras.models import load_model
                except Exception as e:
                    logging.exception('Failed to import TensorFlow/Keras: %s', e)
                    return
                self.model = load_model(model_path)
                self._loaded = True
                logging.info('Keras model loaded from %s', model_path)

            elif fw == 'pytorch' or fw == 'torch':
                try:
                    import torch
                except Exception as e:
                    logging.exception('Failed to import PyTorch: %s', e)
                    return
                # load model - assume entire model was saved (scripted or state_dict)
                try:
                    self.model = torch.load(model_path, map_location=torch.device('cpu'))
                    self.model.eval()
                    self._loaded = True
                    logging.info('PyTorch model loaded from %s', model_path)
                except Exception as e:
                    logging.exception('Failed to load PyTorch model: %s', e)
                    self.model = None

            elif fw == 'onnx' or fw == 'onnxruntime':
                try:
                    import onnxruntime as ort
                except Exception as e:
                    logging.exception('Failed to import onnxruntime: %s', e)
                    return
                try:
                    self.model = ort.InferenceSession(model_path)
                    self._loaded = True
                    logging.info('ONNX model loaded from %s', model_path)
                except Exception as e:
                    logging.exception('Failed to load ONNX model: %s', e)
                    self.model = None

            else:
                logging.warning('Unknown framework "%s" — falling back to placeholder', fw)
        except Exception as e:
            logging.exception('Unexpected error while loading model: %s', e)
            self.model = None

    def preprocess(self, image_bytes):
        try:
            image = Image.open(io.BytesIO(image_bytes))
        except Exception:
            raise InvalidImageError('Cannot open image')

        if self.config.get('color_mode', 'rgb').lower() == 'rgb':
            image = image.convert('RGB')
        elif self.config.get('color_mode', 'grayscale') or self.config.get('color_mode', '').lower() == 'l':
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

        # If channels-first requested by model config, transpose
        if self.config.get('channel_order', 'channels_last') == 'channels_first':
            # from (1, H, W, C) -> (1, C, H, W)
            arr = np.transpose(arr, (0, 3, 1, 2))

        return arr

    def predict_with_placeholder(self, preprocessed):
        # Simple heuristic: average brightness -> dog/cat
        mean_val = float(np.mean(preprocessed))
        confidence = min(max((mean_val - 0.2) * 200, 50), 99.99)
        prediction = 'Dog' if mean_val > 0.5 else 'Cat'
        return {'prediction': prediction, 'confidence': round(confidence, 2)}

    def predict_with_keras(self, preprocessed):
        # Keras expects channels_last by default
        preds = self.model.predict(preprocessed)
        # preds shape can be (1,1) for sigmoid or (1,2+) for softmax
        if preds.ndim == 2 and preds.shape[1] == 1:
            prob = float(preds[0][0])
            label_idx = '1' if prob >= 0.5 else '0'
            confidence = round(prob * 100, 2) if label_idx == '1' else round((1 - prob) * 100, 2)
        else:
            probs = preds[0]
            top_idx = int(np.argmax(probs))
            confidence = round(float(probs[top_idx]) * 100, 2)
            label_idx = str(top_idx)

        label = self.config.get('class_mapping', {}).get(label_idx, label_idx)
        return {'prediction': label, 'confidence': confidence}

    def predict_with_pytorch(self, preprocessed):
        import torch
        # Convert numpy to torch tensor
        tensor = torch.from_numpy(preprocessed)
        # Ensure float32
        tensor = tensor.float()
        # If tensor is channels_last (N,H,W,C) and model expects NCHW, the config should set channel_order
        # Model is assumed to output logits or probabilities
        with torch.no_grad():
            if hasattr(self.model, 'to'):
                # ensure on CPU
                try:
                    self.model.to('cpu')
                except Exception:
                    pass
            # If numpy is channels_last but model expects channels_first, config should have handled transpose
            out = self.model(tensor)
            if isinstance(out, (list, tuple)):
                out = out[0]
            out = out.cpu().numpy()

        if out.ndim == 2 and out.shape[1] == 1:
            prob = float(out[0][0])
            label_idx = '1' if prob >= 0.5 else '0'
            confidence = round(prob * 100, 2) if label_idx == '1' else round((1 - prob) * 100, 2)
        else:
            probs = out[0]
            top_idx = int(np.argmax(probs))
            confidence = round(float(probs[top_idx]) * 100, 2)
            label_idx = str(top_idx)

        label = self.config.get('class_mapping', {}).get(label_idx, label_idx)
        return {'prediction': label, 'confidence': confidence}

    def predict_with_onnx(self, preprocessed):
        # ONNX Runtime expects numpy inputs; find input name
        sess = self.model
        input_name = sess.get_inputs()[0].name
        outputs = sess.run(None, {input_name: preprocessed.astype(np.float32)})
        out = outputs[0]
        if out.ndim == 2 and out.shape[1] == 1:
            prob = float(out[0][0])
            label_idx = '1' if prob >= 0.5 else '0'
            confidence = round(prob * 100, 2) if label_idx == '1' else round((1 - prob) * 100, 2)
        else:
            probs = out[0]
            top_idx = int(np.argmax(probs))
            confidence = round(float(probs[top_idx]) * 100, 2)
            label_idx = str(top_idx)
        label = self.config.get('class_mapping', {}).get(label_idx, label_idx)
        return {'prediction': label, 'confidence': confidence}

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

        if self._loaded and self.framework in ('keras', 'tensorflow'):
            try:
                return self.predict_with_keras(preprocessed)
            except Exception:
                logging.exception('Keras prediction failed — falling back to placeholder')
                return self.predict_with_placeholder(preprocessed)

        elif self._loaded and self.framework in ('pytorch', 'torch'):
            try:
                return self.predict_with_pytorch(preprocessed)
            except Exception:
                logging.exception('PyTorch prediction failed — falling back to placeholder')
                return self.predict_with_placeholder(preprocessed)

        elif self._loaded and self.framework in ('onnx', 'onnxruntime'):
            try:
                return self.predict_with_onnx(preprocessed)
            except Exception:
                logging.exception('ONNX prediction failed — falling back to placeholder')
                return self.predict_with_placeholder(preprocessed)

        else:
            # fallback placeholder
            return self.predict_with_placeholder(preprocessed)
