# Perceptual Similarity in Flax NNX

Port of the perceptual similarity metric (VGG16 only) in Flax's NNX module.
VGG16 weights ported from [torchvision.models](https://pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html#vgg16), linear weights from the original authors [github.com/richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/weights/v0.1/vgg.pth).

# Quick start
Run `pip install lpips-nnx`. The following Python code is all you need.

```python
import jax.numpy as jnp
from lpips_nnx import LPIPS


lpips = LPIPS()
x0 = ...
x1 = ...
xref = ...

d0, d1 = lpips(x0, xref), lpips(x1, xref)
assert d0.item() > d1.item()
```

# Acknowledgements

This repository borrows partially from https://github.com/richzhang/PerceptualSimilarity, https://github.com/pcuenca/lpips-j, and https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/lpips.py. 