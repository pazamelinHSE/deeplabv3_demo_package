# Copyright (c) 2019 Katsuya Hyodo
# https://github.com/PINTO0309/PINTO_model_zoo
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os
import sys
import time
import argparse

import cv2
import numpy as np
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.lite.python.interpreter import Interpreter


def parse_args():
    resources_path = os.path.join(os.path.dirname(__file__), 'resources')
    default_img_path = os.path.join(resources_path, 'img', 'demo.jpg')
    default_model_path = os.path.join(resources_path, 'models', 'deeplab_v3_plus_mnv2_decoder_256_integer_quant.tflite')

    parser = argparse.ArgumentParser()
    parser.add_argument("--deep_model", default=default_model_path, help="Path of the deeplabv3plus model.")
    parser.add_argument("--img", type=str, default=default_img_path, help="Path to the image to process.")
    parser.add_argument("--num_threads", type=int, default=1, help="Threads.")
    args = parser.parse_args()

    return args

def run():
    args = parse_args()
    deep_model  = args.deep_model
    imgpath     = args.img
    num_threads = args.num_threads

    # Get deeplab color palettes
    resources_path = os.path.join(os.path.dirname(__file__), 'resources')
    palette_path = os.path.join(resources_path, 'img', 'colorpalette.png')
    deeplab_palette = Image.open(palette_path).getpalette()

    interpreter = Interpreter(model_path=deep_model)
    try:
        interpreter.set_num_threads(num_threads)
    except:
        print("WARNING: The installed PythonAPI of Tensorflow/Tensorflow Lite runtime does not support Multi-Thread processing.")
        print("WARNING: It works in single thread mode.")
        print("WARNING: If you want to use Multi-Thread to improve performance on aarch64/armv7l platforms, please refer to one of the below to implement a customized Tensorflow/Tensorflow Lite runtime.")
        print("https://github.com/PINTO0309/Tensorflow-bin.git")
        print("https://github.com/PINTO0309/TensorflowLite-bin.git")
        pass
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]['index']
    deeplabv3_predictions = interpreter.get_output_details()[0]['index']
    window_name = 'deeplabv3-plus-demo'

    t1 = time.perf_counter()

    color_image = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)
    image_width, image_height, _ = color_image.shape
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

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
    
    t2 = time.perf_counter()
    print(f'Processing time: {t2 -t1}')

    imdraw = cv2.addWeighted(color_image, 1.0, outputimg, 0.9, 0)
    cv2.imshow(window_name, imdraw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()