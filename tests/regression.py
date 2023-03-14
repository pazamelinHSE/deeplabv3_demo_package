import os
import cv2
import pytest

import deeplabv3_demo
from deeplabv3_demo import demo


def regression_template(img_path: str, number_of_segmented_pixels_expected: int, tolerance_pix=1000):
    input_image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    runner = demo.DemoRunner(demo.DEMO_DEFAULT_MODEL, num_threads=1, silent=True)
    number_of_segmented_pixels_actual = runner.process_image(input_image, save_result=False)
    range_lhs = number_of_segmented_pixels_expected - tolerance_pix
    range_rhs = number_of_segmented_pixels_expected + tolerance_pix
    assert range_lhs <= number_of_segmented_pixels_actual <= range_rhs


def test_regression():
    img = os.path.join(os.path.dirname(__file__), "img", "regression.jpg")
    regression_template(img, number_of_segmented_pixels_expected=694355)
