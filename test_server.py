import argparse
import logging
import base64
import io
import time
import json

from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS
# import neuralgym as ng
from PIL import Image, ImageDraw
import tensorflow as tf
import pycocotools.mask as mask_util

from inpaint_model import InpaintCAModel

logging.basicConfig(
    level=logging.INFO,
    format=('[%(asctime)s] {%(filename)s:%(lineno)d} '
            '%(levelname)s - %(message)s'),
)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default='', type=str,
                        help='The filename of image to be completed.')
    parser.add_argument('--mask', default='', type=str,
                        help='The filename of mask, value 255 indicates mask.')
    parser.add_argument('--output', default='output.png', type=str,
                        help='Where to write output.')
    parser.add_argument('--checkpoint_dir', default='', type=str,
                        help='The directory of tensorflow checkpoint.')
    return parser.parse_args()


sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=sess_config)


class GenerativeInpaintingWorker:

    def __init__(self):
        logger.info("Initializing ..")
        # ng.get_gpus(1)
        self.args = get_args()
        self.model = InpaintCAModel()
        self.grid = 8
        logger.info("Initialization done")

    def _mask_bboxes_to_mask(
        self, image_shape, mask_bboxes, classes, target_classes, threshold
    ):
        try:
            mask = Image.new("RGB", image_shape)
            for i, mask_bbox in enumerate(mask_bboxes):
                if classes[i] in target_classes and mask_bbox[4] > threshold:
                    x1, y1, x2, y2 = mask_bbox
                    mask_shape = (x2 - x1, y2 - y1)
                    mask_fg = Image.new("RGB", mask_shape, (255, 255, 255))
                    mask.paste(mask_fg, (x1, y1))
            mask = np.array(mask)[:, :, ::-1].copy()
        except Exception as err:
            logger.info(err, exc_info=True)
            import pdb
            pdb.set_trace()
            print("")
        return mask

    def _segms_to_mask(
        self, image_shape, mask_bboxes,
        segms, classes, target_classes, threshold
    ):
        try:
            segms = mask_util.decode(segms)
            mask = Image.new("RGB", image_shape)
            for i in range(segms.shape[-1]):
                segm = segms[:, :, i]
                if (classes[i] in target_classes
                        and mask_bboxes[i][4] > threshold):
                    segm = (segm * 255).astype(np.int8)
                    segm = Image.fromarray(segm).convert("RGB")
                    mask.paste(segm, (0, 0))
            mask = np.array(mask)[:, :, ::-1].copy()
        except Exception as err:
            logger.info(err, exc_info=True)
            import pdb
            pdb.set_trace()
            print("")
        return mask

    def _draw_bboxes(self, img, bboxes, th):
        draw = ImageDraw.Draw(img)
        for i, bbox in enumerate(bboxes):
            score = bbox[4]
            x1, y1, x2, y2 = map(int, bbox[:4])
            if score > th:
                draw.rectangle(((x1, y1), (x2, y2)), outline="red")
        return img

    def infer(
        self, image, mask_bboxes, use_mask,
        segms, classes, target_classes, threshold
    ):

        start_time = time.time()
        h, w, _ = image.shape
        logger.info(f"Shape: {image.shape}")

        if not use_mask:
            mask = self._mask_bboxes_to_mask(
                (w, h), mask_bboxes, classes, target_classes, threshold
            )
        else:
            mask = self._segms_to_mask(
                (w, h), mask_bboxes, segms, classes, target_classes, threshold
            )
        assert image.shape == mask.shape, f"{image.shape} vs {mask.shape}"

        image = image[:h//self.grid*self.grid, :w//self.grid*self.grid, :]
        mask = mask[:h//self.grid*self.grid, :w//self.grid*self.grid, :]

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            input_image = tf.constant(input_image, dtype=tf.float32)
            output = self.model.build_server_graph(input_image)
            output = (output + 1.) * 127.5
            output = tf.reverse(output, [-1])
            output = tf.saturate_cast(output, tf.uint8)

            # load pretrained model
            vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            assign_ops = []
            for var in vars_list:
                vname = var.name
                from_name = vname
                var_value = tf.contrib.framework.load_variable(
                    self.args.checkpoint_dir, from_name)
                assign_ops.append(tf.assign(var, var_value))
            sess.run(assign_ops)
            result = sess.run(output)

            im = Image.fromarray(result[0])
            result_image = self._draw_bboxes(im, mask_bboxes, threshold)
            with io.BytesIO() as buf:
                result_image.save(buf, format="jpeg")
                buf.seek(0)
                encoded_string = base64.b64encode(buf.read())
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
        image_file = request.files['pic']
        mask_bboxes = json.loads(request.values['bboxes'])
        use_mask = json.loads(request.values['use_mask'])
        segms = json.loads(request.values['segms'])
        threshold = json.loads(request.values['threshold'])
        classes = json.loads(request.values['classes'])
        # target_classes = json.loads(request.value['target_classes'])
        target_classes = [1]
    except Exception as err:
        logger.error(str(err), exc_info=True)
        raise InvalidUsage(
            f"{err}: request {request} "
            "has no files['raw']"
        )
    if image_file is None or mask_bboxes is None:
        raise InvalidUsage('There is no iamge')
    try:
        image = np.array(
            Image.open(image_file).convert("RGB"))[:, :, ::-1].copy()
    except Exception as err:
        logger.error(str(err), exc_info=True)
        raise InvalidUsage(
            f"{err}: request.files['raw'] {request.files['raw']} "
            "could not be read by opencv"
        )
    try:
        result = gi_worker.infer(
            image, mask_bboxes, use_mask, segms, classes,
            target_classes, threshold
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
