# Copyright (c) 2019 Katsuya Hyodo
# https://github.com/PINTO0309/PINTO_model_zoo
#
# Copyright (c) 2023 Pavel Zamelin
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os
import time
import logging
import argparse

import cv2
import numpy as np
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
try:
    from tflite_runtime.interpreter import Interpreter
except:
    from tensorflow.lite.python.interpreter import Interpreter


DEMO_RESOURCES_PATH = os.path.join(os.path.dirname(__file__), "resources")
DEMO_DEFAULT_IMG = os.path.join(DEMO_RESOURCES_PATH, "img", "demo.jpg")
DEMO_DEFAULT_MODEL = os.path.join(
    DEMO_RESOURCES_PATH,
    "models",
    "deeplab_v3_plus_mnv2_decoder_256_integer_quant.tflite",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deep_model",
        default=DEMO_DEFAULT_MODEL,
        help="Path of the deeplabv3plus model.",
    )
    parser.add_argument(
        "--img",
        type=str,
        default=DEMO_DEFAULT_IMG,
        help="Path to the image to process.",
    )
    parser.add_argument("--num_threads", type=int, default=1, help="Threads.")
    parser.add_argument("--save_result", help="Save the resulting image.", action="store_true")
    parser.add_argument("--silent", help="No logs or image output.", action="store_true")
    args = parser.parse_args()

    return args


class DemoRunner:
    """
    Class that imports a tflite model and runs its inference on provided images
    """

    def __init__(self, deep_model_path: str, num_threads: int, silent: bool):
        # set up logger
        self.silent = silent
        if self.silent:
            logger = logging.getLogger()
            logger.disabled = True

        # load the model into a tflite Interpreter for inference
        self.interpreter = Interpreter(model_path=deep_model_path)
        try:
            self.interpreter.set_num_threads(num_threads)
        except:
            msg = (
                "The installed PythonAPI of Tensorflow/Tensorflow Lite runtime "
                + "does not support Multi-Thread processing."
            )
            logging.warning(msg)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]["index"]
        self.deeplabv3_predictions = self.interpreter.get_output_details()[0]["index"]

        # load deeplab color palettes
        palette_path = os.path.join(DEMO_RESOURCES_PATH, "img", "colorpalette.png")
        self.deeplab_palette = Image.open(palette_path).getpalette()

        # set output window name
        self.window_name = "deeplabv3-plus-demo"

    def _prepare_input_image(self, image):
        # resize, convert and normalize the image to match model input
        prepimg_deep = cv2.resize(image, (256, 256))
        prepimg_deep = cv2.cvtColor(prepimg_deep, cv2.COLOR_BGR2RGB)
        prepimg_deep = np.expand_dims(prepimg_deep, axis=0)
        prepimg_deep = prepimg_deep.astype(np.float32)
        cv2.normalize(prepimg_deep, prepimg_deep, -1, 1, cv2.NORM_MINMAX)
        return prepimg_deep

    def _get_output_image(self, input_image, predictions):
        # get an output image with segmentation results
        image_width, image_height, _ = input_image.shape
        output_image = np.uint8(predictions)
        output_image = cv2.resize(output_image, (image_height, image_width))
        output_image = Image.fromarray(output_image, mode="P")
        output_image.putpalette(self.deeplab_palette)
        output_image = output_image.convert("RGB")
        output_image = np.asarray(output_image)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        output_image_gray = cv2.cvtColor(output_image, cv2.COLOR_RGB2GRAY)
        number_of_segmented_pix = cv2.countNonZero(output_image_gray)
        output_image = cv2.addWeighted(input_image, 1.0, output_image, 0.9, 0)
        return output_image, number_of_segmented_pix

    def get_pallete(self):
        return self.deeplab_palette

    def process_image(self, input_image, save_result: bool):
        time_start = time.perf_counter()

        # prepare input image for the model
        image_width, image_height, _ = input_image.shape
        prepared_image = self._prepare_input_image(input_image)

        # run the model and get resulting tensor
        self.interpreter.set_tensor(self.input_details, prepared_image)
        self.interpreter.invoke()
        predictions = self.interpreter.get_tensor(self.deeplabv3_predictions)[0]

        # get resulting image
        output_image, number_of_segmented_pix = self._get_output_image(input_image, predictions)
        time_end = time.perf_counter()

        if not self.silent:
            if save_result:
                cv2.imwrite("demo-result.jpg", output_image)
            else:
                cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
                cv2.imshow(self.window_name, output_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            print(f"Image dimensions: {image_height}x{image_width}")
            print(f"Processing time: {time_end - time_start}")
            print(f"Number of segmented pixels: {number_of_segmented_pix}")

        return output_image, number_of_segmented_pix


def demo_entrypoint():
    args = parse_args()
    input_image = cv2.imread(args.img, cv2.IMREAD_UNCHANGED)
    runner = DemoRunner(args.deep_model, args.num_threads, args.silent)
    runner.process_image(input_image, args.save_result)


if __name__ == "__main__":
    demo_entrypoint()
