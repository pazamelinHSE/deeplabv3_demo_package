import os
import pytest

import deeplabv3_demo
from deeplabv3_demo import demo


def regression_template(img_path: str, number_of_segmented_pixels_expected: int, tolerance_pix=1000):
    number_of_segmented_pixels_actual = demo.run(
        deep_model=demo.DEMO_DEFAULT_MODEL, img_path=img_path, num_threads=1, save_result=False, silent=True
    )
    range_lhs = number_of_segmented_pixels_expected - tolerance_pix
    range_rhs = number_of_segmented_pixels_expected + tolerance_pix
    assert range_lhs <= number_of_segmented_pixels_actual <= range_rhs


def test_regression():
    img = os.path.join(os.path.dirname(__file__), "img", "regression.jpg")
    regression_template(img, number_of_segmented_pixels_expected=694355)
