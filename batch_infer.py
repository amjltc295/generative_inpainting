import argparse
import logging
import time

import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
import cv2

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

    def _draw_bboxes(self, img, bboxes, th):
        draw = ImageDraw.Draw(img)
        for i, bbox in enumerate(bboxes):
            score = bbox[4]
            x1, y1, x2, y2 = map(int, bbox[:4])
            if score > th:
                draw.rectangle(((x1, y1), (x2, y2)), outline="red")
        return img

    def infer(
        self, images, masks
    ):

        start_time = time.time()
        h, w, _ = images[0].shape
        logger.info(f"Shape: {images.shape}")

        input_images = []
        for image, mask in zip(images, masks):
            image_ = image[:h//self.grid*self.grid, :w//self.grid*self.grid, :]
            mask = mask[:h//self.grid*self.grid, :w//self.grid*self.grid, :]

            image_ = np.expand_dims(image_, 0)
            mask = np.expand_dims(mask, 0)
            input_image = np.concatenate([image_, mask], axis=2)
            input_images.append(input_image)
        input_images = np.vstack(input_images)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            input_images = tf.constant(input_images, dtype=tf.float32)
            outputs = self.model.build_server_graph(input_images)
            outputs = (outputs + 1.) * 127.5
            outputs = tf.reverse(outputs, [-1])
            outputs = tf.saturate_cast(outputs, tf.uint8)

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
            results = sess.run(outputs)

            result_nps = images.copy()
            result_nps[
                :, :h//self.grid*self.grid, :w//self.grid*self.grid, :
            ] = results

            result_images = []
            for result_np in result_nps:
                result_image = Image.fromarray(result_np.astype('uint8'))
                result_images.append(result_image)
            logger.info(f"Time: {time.time() - start_time}")
            return result_images


gi_worker = GenerativeInpaintingWorker()


def batch_infer(images, masks):
    results = gi_worker.infer(images, masks)
    return results


if __name__ == "__main__":

    args = get_args()
    image = np.array(Image.open(args.image).convert("RGB"))
    mask = np.array(Image.open(args.mask).convert("RGB"))
    images = np.array([image/2, image])
    masks = np.array([mask, mask])
    results = batch_infer(images, masks)
    results[0].save(args.output)
    results[1].save(args.output+'_2.png')
