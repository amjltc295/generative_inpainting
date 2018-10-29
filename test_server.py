import argparse
import logging
import json

from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS
# import neuralgym as ng
from PIL import Image

from batch_infer import GenerativeInpaintingWorker


logging.basicConfig(
    level=logging.INFO,
    format=('[%(asctime)s] {%(filename)s:%(lineno)d} '
            '%(levelname)s - %(message)s'),
)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-hh', '--image_height', default=180, type=int,
                        help='Image height')
    parser.add_argument('-ww', '--image_width', default=320, type=int,
                        help='Image width')
    parser.add_argument('--checkpoint_dir',
                        default='model_logs/release_places2_256', type=str,
                        help='The directory of tensorflow checkpoint.')
    return parser.parse_args()


app = Flask(__name__)
CORS(app)


@app.route('/hi', methods=['GET'])
def hi():
    return jsonify(
        {"message": "Hi! This is the generative_inpainting worker (new)."})


@app.route('/gi', methods=['POST'])
def generative_inpainting():
    try:
        image_data = json.loads(request.values['image']).encode('latin-1')
        bboxes = json.loads(request.values['bboxes'])
        image_size = json.loads(request.values['image_size'])
        image_mode = json.loads(request.values['image_mode'])
        image = np.array(
            Image.frombytes(
                image_mode, image_size, image_data
            )
        )
    except Exception as err:
        logger.error(str(err), exc_info=True)
        raise InvalidUsage(
            f"{err}: request {request} "
            "has no files['raw']"
        )
    except Exception as err:
        logger.error(str(err), exc_info=True)
        raise InvalidUsage(
            f"{err}: request.files['raw'] {request.files['raw']} "
            "could not be read by opencv"
        )
    try:
        result = gi_worker.infer(
            np.array([image]), np.array([bboxes])
        )[0]
        result = json.dumps(result.tobytes().decode('latin-1'))
    except Exception as err:
        logger.error(str(err), exc_info=True)
        raise InvalidUsage(
            f"{err}: request {request} "
            "The server encounters some error to process this image",
            status_code=500
        )
    return jsonify({'result': result})


class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


if __name__ == '__main__':
    args = get_args()
    gi_worker = GenerativeInpaintingWorker(
        logger,
        image_height=args.image_height,
        image_width=args.image_width,
        checkpoint_dir=args.checkpoint_dir
    )
    app.run(host='0.0.0.0', port=8083)
