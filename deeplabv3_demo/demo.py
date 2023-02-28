# Copyright (c) 2019 Katsuya Hyodo
# https://github.com/PINTO0309/PINTO_model_zoo
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
    parser.add_argument("--save_result", type=bool, default=False, help="Save the resulting image.")
    parser.add_argument("--silent", help="No logs or image output.", action="store_true")
    args = parser.parse_args()

    return args


def run(deep_model: str, img_path: str, num_threads: int, save_result: bool, silent: bool):
    if silent:
        logger = logging.getLogger()
        logger.disabled = True

    logging.info(f"Processing {img_path} image with {deep_model} model ...")

    # Get deeplab color palettes
    resources_path = os.path.join(os.path.dirname(__file__), "resources")
    palette_path = os.path.join(resources_path, "img", "colorpalette.png")
    deeplab_palette = Image.open(palette_path).getpalette()

    interpreter = Interpreter(model_path=deep_model)
    try:
        interpreter.set_num_threads(num_threads)
    except:
        logging.warning(
            "The installed PythonAPI of Tensorflow/Tensorflow Lite runtime "
            + "does not support Multi-Thread processing."
        )

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]["index"]
    deeplabv3_predictions = interpreter.get_output_details()[0]["index"]
    window_name = "deeplabv3-plus-demo"

    time_start = time.perf_counter()

    color_image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    image_width, image_height, _ = color_image.shape

    # Normalization
    prepimg_deep = cv2.resize(color_image, (256, 256))
    prepimg_deep = cv2.cvtColor(prepimg_deep, cv2.COLOR_BGR2RGB)
    prepimg_deep = np.expand_dims(prepimg_deep, axis=0)
    prepimg_deep = prepimg_deep.astype(np.float32)
    cv2.normalize(prepimg_deep, prepimg_deep, -1, 1, cv2.NORM_MINMAX)

    # Run model - DeeplabV3-plus
    interpreter.set_tensor(input_details, prepimg_deep)
    interpreter.invoke()

    # Get results
    predictions = interpreter.get_tensor(deeplabv3_predictions)[0]

    # Segmentation
    outputimg = np.uint8(predictions)
    outputimg = cv2.resize(outputimg, (image_height, image_width))
    outputimg = Image.fromarray(outputimg, mode="P")
    outputimg.putpalette(deeplab_palette)
    outputimg = outputimg.convert("RGB")
    outputimg = np.asarray(outputimg)
    outputimg = cv2.cvtColor(outputimg, cv2.COLOR_RGB2BGR)

    time_end = time.perf_counter()

    outputimg_gray = cv2.cvtColor(outputimg, cv2.COLOR_RGB2GRAY)
    number_of_segmented_pix = cv2.countNonZero(outputimg_gray)

    imdraw = cv2.addWeighted(color_image, 1.0, outputimg, 0.9, 0)
    if not silent:
        if save_result:
            cv2.imwrite("demo-result.jpg", imdraw)
        else:
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(window_name, imdraw)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        print("")
        print(f"Image dimensions: {image_height}x{image_width}")
        print(f"Processing time: {time_end - time_start}")
        print(f"Number of segmented pixels: {number_of_segmented_pix}")

    return number_of_segmented_pix


def demo_entrypoint():
    args = parse_args()
    run(
        deep_model=args.deep_model,
        img_path=args.img,
        num_threads=args.num_threads,
        save_result=args.save_result,
        silent=args.silent,
    )


if __name__ == "__main__":
    demo_entrypoint()
