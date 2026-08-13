import io
from PIL import Image
import pytest

from app import app as flask_app


@pytest.fixture
def client():
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client


def create_test_image(format='JPEG', color=(255, 0, 0)):
    img = Image.new('RGB', (100, 100), color=color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    return buf


def test_index(client):
    rv = client.get('/')
    assert rv.status_code == 200
    assert b'Dog & Cat Image Classifier' in rv.data


def test_predict_valid_jpeg(client):
    img = create_test_image('JPEG')
    data = {'image': (img, 'test.jpg')}
    rv = client.post('/predict', data=data, content_type='multipart/form-data')
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert 'success' in json_data
    assert json_data['success'] is True
    assert 'prediction' in json_data
    assert 'confidence' in json_data


def test_predict_invalid_file_type(client):
    buf = io.BytesIO(b'not-an-image')
    data = {'image': (buf, 'test.txt')}
    rv = client.post('/predict', data=data, content_type='multipart/form-data')
    assert rv.status_code == 400
    json_data = rv.get_json()
    assert json_data['success'] is False
