import base64
import json
from pathlib import Path

import pytest


@pytest.fixture
def data_dir() -> Path:
    return Path(__file__).parent / "data"


@pytest.fixture
def conversation1(data_dir: Path) -> list[dict[str, str]]:
    with open(data_dir / "conversation1.json") as f:
        return json.load(f)


@pytest.fixture
def conversation2(data_dir: Path) -> list[dict[str, str]]:
    with open(data_dir / "conversation2.json") as f:
        return json.load(f)


@pytest.fixture
def base64_sunglasses_image(data_dir: Path) -> str:
    with open(data_dir / "sunglasses.png", "rb") as f:
        image = f.read()
    base64_image = base64.b64encode(image).decode("utf-8")
    return f"data:image/png;base64,{base64_image}"
