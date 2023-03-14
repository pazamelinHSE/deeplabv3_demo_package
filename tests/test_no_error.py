import os
import cv2
import pytest

import deeplabv3_demo
from deeplabv3_demo import demo


def no_error_template(img_path: str):
    input_image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    runner = demo.DemoRunner(demo.DEMO_DEFAULT_MODEL, num_threads=1, silent=True)
    output_image, number_of_segmented_pixels_actual = runner.process_image(input_image, save_result=False)
    assert number_of_segmented_pixels_actual


def test_no_error_small():
    img = os.path.join(os.path.dirname(__file__), "img", "small.jpg")
    no_error_template(img)


def test_no_error_large():
    img = os.path.join(os.path.dirname(__file__), "img", "large.jpg")
    no_error_template(img)
