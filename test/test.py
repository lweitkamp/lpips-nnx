import numpy as np
import pytest
from PIL import Image

import torch

from lpips_nnx import LPIPS as LPIPS_NNX
from lpips import LPIPS


def preprocess_image(image: Image) -> np.ndarray:
    return 2 * (np.array(image, dtype=np.float32) / 255.0) - 1

def torchify(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image.T)


@pytest.fixture
def images():
    images = [
        preprocess_image(Image.open(image))
        for image in ("test/ex_p0.png", "test/ex_p1.png", "test/ex_ref.png")
    ]
    return images

@pytest.fixture
def lpips_from_nnx() -> LPIPS_NNX:
    return LPIPS_NNX()

@pytest.fixture
def lpips_torch() -> LPIPS:
    return LPIPS(net="vgg")


def test_distance(images, lpips_torch, lpips_from_nnx):
    """Assert that the distance between p0 and ref is greater than the distance between p1 and ref."""
    x0, x1, ref = images
    
    d0 = lpips_from_nnx(x0, ref).item()
    d1 = lpips_from_nnx(x1, ref).item()
    assert d0 > d1
    
    d0 = lpips_torch(torchify(x0), torchify(ref)).item()
    d1 = lpips_torch(torchify(x1), torchify(ref)).item()
    assert d0 > d1


def test_close(images, lpips_torch, lpips_from_nnx):
    """Assert that the LPIPS package and LPIPS-NNX package return similar results."""
    x0, x1, ref = images
    
    for x in (x0, x1):
        d0 = lpips_from_nnx(x, ref).item()
        d1 = lpips_torch(torchify(x), torchify(ref)).item()
        assert np.isclose(d0, d1, atol=1e-5)
