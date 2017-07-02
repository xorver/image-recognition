import os

from flask import Flask
from flask import request
from  imagenet import classify_image

app = Flask(__name__)


class Config(object):
    MODEL_DIR = '/tmp/imagenet'
    UPLOAD_FOLDER = '/tmp'


@app.route('/api/v1.0/recognize', methods=['POST'])
def recognize():
    file = request.files['image']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    categories = classify_image.run_inference_on_image(file_path)
    return categories[0], 200


if __name__ == '__main__':
    app.config.from_object('image-recognition.Config')
    classify_image.set_model_dir(app.config['MODEL_DIR'])
    classify_image.maybe_download_and_extract()
    classify_image.create_graph()
    app.run()
