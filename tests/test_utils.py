import pytest

from concurrent_openai.utils import get_png_dimensions, num_tokens_for_image


@pytest.mark.parametrize(
    "width, height, low_resolution, expected",
    [
        (100, 100, True, 85),
        (1000, 1000, True, 85),
        (128, 128, False, 255),
        (256, 128, False, 255),
        (512, 128, False, 255),
        (513, 128, False, 425),
        (513, 513, False, 765),
        (760, 760, False, 765),
        (768, 768, False, 765),
        (1024, 1024, False, 765),
        (1530, 768, False, 1105),
        (1900, 1000, False, 1105),
        (2049, 784, False, 1445),
        (2100, 784, False, 1445),
        (3100, 784, False, 1445),
        (5000, 784, False, 765),
        (1900, 5000, False, 1445),
    ],
)
def test_num_tokens_for_image(width, height, low_resolution, expected):
    assert num_tokens_for_image(width, height, low_resolution) == expected


def test_get_png_dimensions(base64_sunglasses_image):
    width, height = get_png_dimensions(base64_sunglasses_image)
    assert width == height == 20
