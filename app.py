from flask import Flask, render_template, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
from prediction import Predictor, InvalidImageError

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Initialize predictor (loads model if present)
predictor = Predictor()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image part in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Unsupported file type'}), 400

    filename = secure_filename(file.filename)

    try:
        # We keep file in memory via file.stream by passing file to predictor
        result = predictor.predict_image(file)
        return jsonify({'success': True, 'prediction': result['prediction'], 'confidence': result['confidence']}), 200
    except InvalidImageError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        current_app.logger.exception('Prediction error')
        return jsonify({'success': False, 'error': 'Prediction failed'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
