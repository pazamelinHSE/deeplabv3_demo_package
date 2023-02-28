import os
import pytest

import deeplabv3_demo
from deeplabv3_demo import demo


def no_error_template(img_path: str):
    result = demo.run(
        deep_model=demo.DEMO_DEFAULT_MODEL, img_path=img_path, num_threads=1, save_result=False, silent=True
    )
    assert result


def test_no_error_small():
    img = os.path.join(os.path.dirname(__file__), "img", "small.jpg")
    no_error_template(img)


def test_no_error_large():
    img = os.path.join(os.path.dirname(__file__), "img", "large.jpg")
    no_error_template(img)
