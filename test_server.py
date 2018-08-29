import argparse
import logging
import base64
import time

from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS
import neuralgym as ng
import cv2
import tensorflow as tf

from inpaint_model import InpaintCAModel

logging.basicConfig(
    level=logging.INFO,
    format=('[%(asctime)s] {%(filename)s:%(lineno)d} '
            '%(levelname)s - %(message)s'),
)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', default='', type=str,
                        help='The directory of tensorflow checkpoint.')
    return parser.parse_args()


class GenerativeInpaintingWorker:

    def __init__(self):
        logger.info("Initializing ..")
        ng.get_gpus(1)
        self.args = get_args()
        self.model = InpaintCAModel()
        self.grid = 8
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self._load_model()
        logger.info("Initialization done")

    def _load_model(self):
        with self.sess:
            input_placeholder = tf.placeholder(tf.float32)
            self.model.build_server_graph(input_placeholder)
            # load pretrained model
            vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            assign_ops = []
            for var in vars_list:
                vname = var.name
                from_name = vname
                var_value = tf.contrib.framework.load_variable(
                    self.args.checkpoint_dir, from_name)
                assign_ops.append(tf.assign(var, var_value))
            self.sess.run(assign_ops)
            logger.info('Model loaded.')

    def infer(self, image, mask):

        start_time = time.time()

        assert image.shape == mask.shape

        h, w, _ = image.shape
        image = image[:h//self.grid*self.grid, :w//self.grid*self.grid, :]
        mask = mask[:h//self.grid*self.grid, :w//self.grid*self.grid, :]

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)

        input_image = np.concatenate([image, mask], axis=2)
        input_image = tf.constant(input_image, dtype=tf.float32)
        output = self.model.build_server_graph(input_image)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)

        result = self.sess.run(output)
        buf = cv2.imencode(".jpg", result[0][:, :, ::-1])[1]
        encoded_string = base64.b64encode(buf)
        encoded_result_image = (
            b'data:image/jpeg;base64,' + encoded_string
        )
        logger.info("Infer time: {}".format(time.time() - start_time))
        return encoded_result_image


app = Flask(__name__)
CORS(app)
gi_worker = GenerativeInpaintingWorker()


@app.route('/hi', methods=['GET'])
def hi():
    return jsonify(
        {"message": "Hi! This is the generative_inpainting worker."})


@app.route('/gi', methods=['POST'])
def generative_inpainting():
    try:
        image_file = request.files['image']
        mask_file = request.files['mask']
    except Exception as err:
        logger.error(str(err), exc_info=True)
        raise InvalidUsage(
            f"{err}: request {request} "
            "has no files['raw']"
        )
    if image_file is None or mask_file is None:
        raise InvalidUsage('There is no iamge')
    try:
        image = cv2.imread(image_file)
        mask = cv2.imread(mask_file)
    except Exception as err:
        logger.error(str(err), exc_info=True)
        raise InvalidUsage(
            f"{err}: request.files['raw'] {request.files['raw']} "
            "could not be read by opencv"
        )
    try:
        result = gi_worker.infer(
            image, mask
        )
    except Exception as err:
        logger.error(str(err), exc_info=True)
        raise InvalidUsage(
            f"{err}: request {request} "
            "The server encounters some error to process this image",
            status_code=500
        )
    return jsonify({'result': result.decode('utf-8')})


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
    app.run(host='0.0.0.0', port=8083)
